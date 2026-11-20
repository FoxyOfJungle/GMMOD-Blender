bl_info = {
    "name": "GameMaker 8 Model Format (.gmmod/.d3d)",
    "author": "Mozart Junior (@foxyofjungle)",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "File > Import/Export > GameMaker 8 Model (.gmmod/.d3d)",
    "description": "Import and export GameMaker 8 3D models.",
    "category": "Import-Export",
}

import os

import bpy
import numpy as np
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    FloatProperty,
    StringProperty,
)
from bpy_extras.io_utils import ExportHelper, ImportHelper, poll_file_object_drop

GM_FORMAT_VERSION = 100

CMD_PRIMITIVE_BEGIN = 0
CMD_PRIMITIVE_END = 1
CMD_VERTEX_FIRST = 2
CMD_VERTEX_LAST = 9
CMD_SHAPE_FIRST = 10
CMD_SHAPE_LAST = 15

PR_POINTLIST = 1
PR_LINELIST = 2
PR_LINESTRIP = 3
PR_TRIANGLELIST = 4
PR_TRIANGLESTRIP = 5
PR_TRIANGLEFAN = 6

SHAPE_NAMES = {
    10: "block",
    11: "cylinder",
    12: "cone",
    13: "ellipsoid",
    14: "wall",
    15: "floor",
}

# d3d_model_save always pads the arguments up to 10 with zeros.
ARG_COUNT = 10


class GMModelError(Exception):
    pass


#region //////////////////////////// UTILS ////////////////////////////////////

def _srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _num(token, lineno):
    try:
        return float(token)
    except ValueError:
        pass
    try:
        return float(token.replace(",", "."))
    except ValueError:
        raise GMModelError("Line %d: invalid numeric value '%s'" % (lineno, token))


def _set_uv(layer, flat):
    try:
        layer.uv.foreach_set("vector", flat)
    except (AttributeError, RuntimeError, TypeError):
        layer.data.foreach_set("uv", flat)


def _get_uv(layer, flat):
    try:
        layer.uv.foreach_get("vector", flat)
    except (AttributeError, RuntimeError, TypeError):
        layer.data.foreach_get("uv", flat)


def _ensure_object_mode():
    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')
    except RuntimeError:
        pass

#endregion


#region /////////////////////////// FORMAT CLASS //////////////////////////////

class GMModel:
    """Raw contents of a GameMaker model file. Each vertex is a tuple (pos, normal|None, uv|None, (color, alpha)|None)."""

    def __init__(self):
        self.triangles = []
        self.lines = []
        self.points = []
        self.has_normals = False
        self.has_uvs = False
        self.has_colors = False
        self.shapes_skipped = {}
        self.unknown_commands = 0
        self.stray_vertices = 0
        self.declared_count = 0
        self.entries_read = 0
        self.version = GM_FORMAT_VERSION

class __ObjectData:
    __slots__ = ("pos", "nor", "uv", "col", "alpha")

#endregion


#region //////////////////////////// READ MODEL ///////////////////////////////

def build_mesh(model, name, scale=1.0, flip_v=True, flip_winding=False, merge=True, srgb=True, use_normals=True):
    vmap = {}
    co = []

    def vid(pos, always_merge=False):
        if merge or always_merge:
            idx = vmap.get(pos)
            if idx is not None:
                return idx
            vmap[pos] = len(co)
        co.append((pos[0] * scale, pos[1] * scale, pos[2] * scale))
        return len(co) - 1

    faces = []
    loop_nor = []
    loop_uv = []
    loop_col = []
    degenerate = 0

    for tri in model.triangles:
        if flip_winding:
            tri = (tri[0], tri[2], tri[1])
        ids = (vid(tri[0][0]), vid(tri[1][0]), vid(tri[2][0]))
        if merge and (ids[0] == ids[1] or ids[1] == ids[2] or ids[0] == ids[2]):
            degenerate += 1
            continue
        faces.append(ids)
        for _pos, nor, uv, col in tri:
            if model.has_normals:
                loop_nor.append(nor if nor is not None else (0.0, 0.0, 0.0))
            if model.has_uvs:
                if uv is None:
                    loop_uv.append((0.0, 0.0))
                else:
                    loop_uv.append((uv[0], 1.0 - uv[1] if flip_v else uv[1]))
            if model.has_colors:
                if col is None:
                    loop_col.append((16777215, 1.0))
                else:
                    loop_col.append((int(round(col[0])), col[1]))

    edges = []
    for a, b in model.lines:
        ia, ib = vid(a[0], True), vid(b[0], True)
        if ia != ib:
            edges.append((ia, ib))
    for p in model.points:
        vid(p[0], True)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(co, edges, faces)
    nloops = len(mesh.loops)
    if nloops != len(faces) * 3:
        raise GMModelError("Internal error: unexpected number of loops.")

    if nloops:
        if model.has_uvs:
            layer = mesh.uv_layers.new(name="UVMap")
            _set_uv(layer, np.asarray(loop_uv, dtype=np.float32).ravel())

        if model.has_colors:
            arr = np.asarray(loop_col, dtype=np.float64)
            ci = arr[:, 0].astype(np.int64)
            rgb = np.stack([ci & 255, (ci >> 8) & 255, (ci >> 16) & 255], axis=1) / 255.0
            if srgb:
                rgb = _srgb_to_linear(rgb)
            rgba = np.concatenate([rgb, np.clip(arr[:, 1:2], 0.0, 1.0)], axis=1)
            attr = mesh.color_attributes.new("Color", 'FLOAT_COLOR', 'CORNER')
            attr.data.foreach_set("color", rgba.astype(np.float32).ravel())
            try:
                mesh.color_attributes.active_color = attr
                mesh.color_attributes.render_color_index = (
                    mesh.color_attributes.find(attr.name))
            except (AttributeError, RuntimeError):
                pass

    mesh.update()

    if nloops and model.has_normals and use_normals:
        nor = np.asarray(loop_nor, dtype=np.float64)
        length = np.linalg.norm(nor, axis=1, keepdims=True)
        nor = np.divide(nor, length, out=np.zeros_like(nor), where=length > 1e-12)
        mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
        mesh.normals_split_custom_set(nor.tolist())
        mesh.update()

    return mesh, degenerate

def read_gmmod(filepath):
    with open(filepath, "r", encoding="latin-1", newline="") as f:
        text = f.read()

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [(n, ln.split()) for n, ln in enumerate(text.split("\n"), 1) if ln.strip()]
    if len(lines) < 2:
        raise GMModelError("File is empty or is not a GameMaker model.")

    model = GMModel()
    model.version = int(_num(lines[0][1][0], lines[0][0]))
    model.declared_count = int(_num(lines[1][1][0], lines[1][0]))

    kind = None
    verts = []

    def flush():
        nonlocal kind, verts
        if kind is None:
            return
        v = verts
        n = len(v)
        if kind == PR_TRIANGLELIST:
            for i in range(0, n - 2, 3):
                model.triangles.append((v[i], v[i + 1], v[i + 2]))
        elif kind == PR_TRIANGLESTRIP:
            for i in range(n - 2):
                if i % 2 == 0:
                    model.triangles.append((v[i], v[i + 1], v[i + 2]))
                else:  # keeps the face orientation
                    model.triangles.append((v[i + 1], v[i], v[i + 2]))
        elif kind == PR_TRIANGLEFAN:
            for i in range(1, n - 1):
                model.triangles.append((v[0], v[i], v[i + 1]))
        elif kind == PR_LINELIST:
            for i in range(0, n - 1, 2):
                model.lines.append((v[i], v[i + 1]))
        elif kind == PR_LINESTRIP:
            for i in range(n - 1):
                model.lines.append((v[i], v[i + 1]))
        elif kind == PR_POINTLIST:
            model.points.extend(v)
        kind = None
        verts = []

    entries = lines[2:2 + model.declared_count]
    for lineno, toks in entries:
        try:
            cmd = int(_num(toks[0], lineno))
        except GMModelError:
            break  # junk after the real commands
        args = [_num(t, lineno) for t in toks[1:]]
        if len(args) < ARG_COUNT:
            args += [0.0] * (ARG_COUNT - len(args))
        model.entries_read += 1

        if cmd == CMD_PRIMITIVE_BEGIN:
            flush()
            kind = int(args[0])
            verts = []
        elif cmd == CMD_PRIMITIVE_END:
            flush()
        elif CMD_VERTEX_FIRST <= cmd <= CMD_VERTEX_LAST:
            flags = cmd - CMD_VERTEX_FIRST
            has_col = bool(flags & 1)
            has_tex = bool(flags & 2)
            has_nor = bool(flags & 4)

            pos = (args[0], args[1], args[2])
            i = 3
            nor = uv = col = None
            if has_nor:
                nor = (args[i], args[i + 1], args[i + 2])
                i += 3
                model.has_normals = True
            if has_tex:
                uv = (args[i], args[i + 1])
                i += 2
                model.has_uvs = True
            if has_col:
                col = (args[i], args[i + 1])
                i += 2
                model.has_colors = True

            if kind is None:
                model.stray_vertices += 1
            else:
                verts.append((pos, nor, uv, col))
        elif CMD_SHAPE_FIRST <= cmd <= CMD_SHAPE_LAST:
            name = SHAPE_NAMES[cmd]
            model.shapes_skipped[name] = model.shapes_skipped.get(name, 0) + 1
        else:
            model.unknown_commands += 1

    flush()  # primitive without primitive_end
    return model

#endregion


#region /////////////////////////// WRITE MODFL ///////////////////////////////

def _extract_object(obj, depsgraph, opts):
    """Returns __ObjectData (per-triangle-vertex arrays) or None."""
    source = obj.evaluated_get(depsgraph) if opts["modifiers"] else obj.original
    try:
        mesh = source.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    except RuntimeError:
        return None
    if mesh is None:
        return None

    try:
        mesh.calc_loop_triangles()
        ntri = len(mesh.loop_triangles)
        if ntri == 0:
            return None

        nloops = len(mesh.loops)
        nverts = len(mesh.vertices)

        tri_loops = np.empty(ntri * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", tri_loops)
        tri_loops = tri_loops.reshape(-1, 3)

        loop_vert = np.empty(nloops, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_vert)

        co = np.empty(nverts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(np.float64)

        # Transform (global scale + object matrix)
        mat = np.array(obj.matrix_world if opts["transform"] else np.identity(4),
                       dtype=np.float64)
        mat[:3, :] *= opts["scale"]
        m3 = mat[:3, :3]
        flip = opts["flip_winding"]
        if np.linalg.det(m3) < 0.0:
            flip = not flip  # mirroring reverses the vertex order
        if flip:
            tri_loops = tri_loops[:, [0, 2, 1]]
        tri_loops = np.ascontiguousarray(tri_loops).ravel()
        tri_verts = loop_vert[tri_loops]

        d = __ObjectData()
        d.pos = co[tri_verts] @ m3.T + mat[:3, 3]

        # Normals
        d.nor = None
        if opts["normals"]:
            nor = np.empty(nloops * 3, dtype=np.float32)
            mesh.corner_normals.foreach_get("vector", nor)
            nor = nor.reshape(-1, 3).astype(np.float64)[tri_loops]
            try:
                nm = np.linalg.inv(m3).T
            except np.linalg.LinAlgError:
                nm = m3
            nor = nor @ nm.T
            length = np.linalg.norm(nor, axis=1, keepdims=True)
            d.nor = np.divide(nor, length, out=np.zeros_like(nor), where=length > 1e-12)

        # UV
        d.uv = None
        if opts["uvs"] and len(mesh.uv_layers):
            layer = mesh.uv_layers.active or mesh.uv_layers[0]
            uv = np.empty(nloops * 2, dtype=np.float32)
            _get_uv(layer, uv)
            uv = uv.reshape(-1, 2).astype(np.float64)[tri_loops]
            if opts["flip_v"]:
                uv[:, 1] = 1.0 - uv[:, 1]
            d.uv = uv

        # Color
        d.col = None
        d.alpha = None
        if opts["colors"] and len(mesh.color_attributes):
            attrs = mesh.color_attributes
            attr = getattr(attrs, "active_color", None) or attrs[0]
            if attr.domain in {'POINT', 'CORNER'}:
                n = len(attr.data)
                rgba = np.empty(n * 4, dtype=np.float32)
                attr.data.foreach_get("color", rgba)
                rgba = rgba.reshape(-1, 4).astype(np.float64)
                rgba = rgba[tri_verts] if attr.domain == 'POINT' else rgba[tri_loops]
                rgb = rgba[:, :3]
                if opts["srgb"]:
                    rgb = _linear_to_srgb(rgb)
                rgb = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.int64)
                d.col = rgb[:, 0] | (rgb[:, 1] << 8) | (rgb[:, 2] << 16)
                d.alpha = np.clip(rgba[:, 3], 0.0, 1.0)
        return d
    finally:
        source.to_mesh_clear()

def write_gmmod(filepath, chunks, use_normals, use_uvs, use_colors):
    """Writes the data of several objects into a single primitive (trianglelist)."""
    total = sum(len(c.pos) for c in chunks)

    cmd = CMD_VERTEX_FIRST + (1 if use_colors else 0) + (2 if use_uvs else 0) \
        + (4 if use_normals else 0)

    fields = ["%.6f"] * 3
    if use_normals:
        fields += ["%.6f"] * 3
    if use_uvs:
        fields += ["%.6f"] * 2
    if use_colors:
        fields += ["%d", "%.6f"]
    pad = ARG_COUNT - len(fields)
    fmt = "%d " + " ".join(fields) + " 0.000000" * pad

    out = [str(GM_FORMAT_VERSION), str(total + 2)]
    out.append("%d %d" % (CMD_PRIMITIVE_BEGIN, PR_TRIANGLELIST) + " 0" * (ARG_COUNT - 1))

    for c in chunks:
        n = len(c.pos)
        cols = [c.pos]
        if use_normals:
            cols.append(c.nor if c.nor is not None else np.zeros((n, 3)))
        if use_uvs:
            cols.append(c.uv if c.uv is not None else np.zeros((n, 2)))
        if use_colors:
            if c.col is None:
                color = np.full(n, 16777215, dtype=np.int64)
                alpha = np.ones(n)
            else:
                color, alpha = c.col, c.alpha
        # Built line by line (mixed types rule out np.savetxt)
        floats = np.concatenate(cols, axis=1).tolist()
        if use_colors:
            color = color.tolist()
            alpha = alpha.tolist()
            for row, ci, al in zip(floats, color, alpha):
                out.append(fmt % ((cmd, *row, ci, al)))
        else:
            for row in floats:
                out.append(fmt % ((cmd, *row)))

    out.append("%d" % CMD_PRIMITIVE_END + " 0" * ARG_COUNT)
    out.append("")

    with open(filepath, "w", encoding="ascii", newline="") as f:
        f.write("\r\n".join(out))
    return total // 3

#endregion


#region /////////////////////////// OPERAYORS /////////////////////////////////

class IMPORT_OT_gmmod(bpy.types.Operator, ImportHelper):
    """Import a GameMaker 8 3D model"""
    bl_idname = "import_scene.gmmod"
    bl_label = "Import GameMaker 8 Model"
    bl_options = {'UNDO', 'PRESET'}

    filename_ext = ".gmmod"
    filter_glob: StringProperty(default="*.gmmod;*.d3d;*.mod", options={'HIDDEN'})

    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={'HIDDEN', 'SKIP_SAVE'})
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})

    global_scale: FloatProperty(name="Scale", default=1.0, min=1e-6, max=1e6, description="Multiplies the model coordinates")
    merge_vertices: BoolProperty(name="Merge Vertices", default=True, description="Merges vertices at the same position (GameMaker models store 3 loose vertices per triangle). Normals, UVs and colors stay per corner, so nothing is lost")
    use_normals: BoolProperty(name="Normals", default=True, description="Applies the normals from the file as custom normals")
    flip_v: BoolProperty(name="Flip V", default=True, description="Flips the V axis of the texture (GameMaker: (0,0) is the top-left corner; Blender: bottom-left)")
    flip_winding: BoolProperty(name="Flip Winding Order", default=False, description="Reverses the vertex order of every triangle")
    color_srgb: BoolProperty(name="Colors as sRGB", default=True, description="Treats the GameMaker color as sRGB and converts it to linear (Blender stores color attributes in linear). This way the color seen in Blender matches the one seen in the game")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(self, "global_scale")
        layout.prop(self, "merge_vertices")
        layout.prop(self, "use_normals")
        layout.prop(self, "flip_v")
        layout.prop(self, "flip_winding")
        layout.prop(self, "color_srgb")

    def execute(self, context):
        if self.files:
            paths = [os.path.join(self.directory, f.name) for f in self.files]
        else:
            paths = [self.filepath]

        _ensure_object_mode()
        for o in list(context.selected_objects):
            o.select_set(False)

        imported = 0
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                model = read_gmmod(path)
                if not (model.triangles or model.lines or model.points):
                    raise GMModelError("No geometry found in '%s'." % name)
                mesh, degenerate = build_mesh(
                    model, name,
                    scale=self.global_scale,
                    flip_v=self.flip_v,
                    flip_winding=self.flip_winding,
                    merge=self.merge_vertices,
                    srgb=self.color_srgb,
                    use_normals=self.use_normals,
                )
            except (GMModelError, OSError) as e:
                self.report({'ERROR'}, "%s: %s" % (os.path.basename(path), e))
                continue
            
            obj = bpy.data.objects.new(name, mesh)
            context.collection.objects.link(obj)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            imported += 1
            
            self.report({'INFO'}, "%s: %d vertices, %d triangles" % (name, len(mesh.vertices), len(mesh.polygons)))
            if model.version != GM_FORMAT_VERSION:
                self.report({'WARNING'}, "%s: version %d (expected 100)" % (name, model.version))
            if degenerate:
                self.report({'INFO'}, "%s: %d degenerate triangles ignored" % (name, degenerate))
            if model.shapes_skipped:
                skipped = ", ".join("%s x%d" % kv for kv in model.shapes_skipped.items())
                self.report({'WARNING'}, "%s: shapes ignored (%s). Only vertex primitives are imported." % (name, skipped))
            if model.unknown_commands:
                self.report({'WARNING'}, "%s: %d unknown commands ignored" % (name, model.unknown_commands))

        return {'FINISHED'} if imported else {'CANCELLED'}


class EXPORT_OT_gmmod(bpy.types.Operator, ExportHelper):
    """Export to the GameMaker 8 3D model format (d3d_model_load)"""
    bl_idname = "export_scene.gmmod"
    bl_label = "Export GameMaker 8 Model"
    bl_options = {'UNDO', 'PRESET'}

    filename_ext = ".gmmod"
    filter_glob: StringProperty(default="*.gmmod;*.d3d;*.mod", options={'HIDDEN'})

    use_selection: BoolProperty(name="Selection Only", default=True, description="Exports only the selected objects")
    use_modifiers: BoolProperty(name="Apply Modifiers", default=True, description="Applies modifiers (as shown in the viewport)")
    apply_transform: BoolProperty(name="Apply Object Transform", default=True, description="Exports in world coordinates (object location, rotation and scale)")
    global_scale: FloatProperty(name="Scale", default=1.0, min=1e-6, max=1e6, description="Multiplies the exported coordinates")
    use_normals: BoolProperty(name="Normals", default=True, description="Exports per-vertex normals (uses the final normals, including smooth/flat shading and custom normals)")
    use_uvs: BoolProperty(name="Texture Coordinates", default=True, description="Exports the active UV map")
    use_colors: BoolProperty(name="Vertex Color + Alpha", default=True, description="Exports the active color attribute (RGBA). Ignored if no object has a color attribute")
    flip_v: BoolProperty(name="Flip V", default=True, description="Flips the V axis of the texture (GameMaker: (0,0) is the top-left corner)")
    flip_winding: BoolProperty(name="Flip Winding Order", default=False, description="Reverses the vertex order of every triangle")
    color_srgb: BoolProperty(name="Colors as sRGB", default=True, description="Converts the color from linear (Blender) to sRGB before writing the GameMaker BGR integer")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.label(text="Objects")
        layout.prop(self, "use_selection")
        layout.prop(self, "use_modifiers")
        layout.prop(self, "apply_transform")
        layout.prop(self, "global_scale")
        layout.separator()
        layout.label(text="Vertex Data")
        layout.prop(self, "use_normals")
        layout.prop(self, "use_uvs")
        layout.prop(self, "use_colors")
        layout.separator()
        layout.prop(self, "flip_v")
        layout.prop(self, "flip_winding")
        layout.prop(self, "color_srgb")

    def execute(self, context):
        _ensure_object_mode()

        candidates = context.selected_objects if self.use_selection else context.visible_objects
        valid_types = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}
        objects = [o for o in candidates if o.type in valid_types]
        if not objects:
            self.report({'ERROR'}, "No mesh objects to export.")
            return {'CANCELLED'}

        opts = {
            "modifiers": self.use_modifiers,
            "transform": self.apply_transform,
            "scale": self.global_scale,
            "normals": self.use_normals,
            "uvs": self.use_uvs,
            "colors": self.use_colors,
            "flip_v": self.flip_v,
            "flip_winding": self.flip_winding,
            "srgb": self.color_srgb,
        }

        depsgraph = context.evaluated_depsgraph_get()
        chunks = []
        for obj in objects:
            data = _extract_object(obj, depsgraph, opts)
            if data is not None:
                chunks.append(data)
        if not chunks:
            self.report({'ERROR'}, "The objects have no faces to export.")
            return {'CANCELLED'}

        has_uv = self.use_uvs and any(c.uv is not None for c in chunks)
        has_col = self.use_colors and any(c.col is not None for c in chunks)

        try:
            ntri = write_gmmod(self.filepath, chunks, self.use_normals, has_uv, has_col)
        except OSError as e:
            self.report({'ERROR'}, "Could not write the file: %s" % e)
            return {'CANCELLED'}

        self.report({'INFO'}, "Exported: %d triangles (%d vertices) from %d object(s)" % (
            ntri, ntri * 3, len(chunks)))
        if self.use_uvs and not has_uv:
            self.report({'WARNING'}, "No UV map found: texture coordinates omitted.")
        if self.use_colors and not has_col:
            self.report({'INFO'}, "No color attribute found: colors omitted.")
        return {'FINISHED'}


class IO_FH_gmmod(bpy.types.FileHandler):
    bl_idname = "IO_FH_gmmod"
    bl_label = "GameMaker 8 Model"
    bl_import_operator = "import_scene.gmmod"
    bl_export_operator = "export_scene.gmmod"
    bl_file_extensions = ".gmmod;.d3d"

    @classmethod
    def poll_drop(cls, context):
        return poll_file_object_drop(context)

#endregion


#region /////////////////////////// REGISTER //////////////////////////////////

def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_gmmod.bl_idname, text="GameMaker 8 Model (.gmmod/.d3d)")

def menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_gmmod.bl_idname, text="GameMaker 8 Model (.gmmod/.d3d)")

classes = (
    IMPORT_OT_gmmod,
    EXPORT_OT_gmmod,
    IO_FH_gmmod,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()

#endregion
