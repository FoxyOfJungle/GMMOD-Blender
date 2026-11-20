
# GameMaker 8 Model Importer/Exporter for Blender

A Blender add-on that imports and exports the 3D model format used by GameMaker 8, the one you load with `d3d_model_load()` and save with `d3d_model_save()`. The files usually end in `.gmmod` or `.d3d`.

It reads and writes vertex positions, normals, texture coordinates and vertex color with alpha.

> For Blender 5.2 or newer.


## Install

1. Download `gmmod_loader.py`.
2. Copy it into Blender's `scripts/addons` folder, or use *Install from Disk* in *Edit > Preferences > Add-ons*.
3. Enable **GameMaker 8 Model Format (.gmmod)**.


## Use

Go to **File > Import** or **File > Export** and choose *GameMaker 8 Model (.gmmod)*. You can also drag a model file into the Blender window to import it.


## Things worth knowing

- The V texture coordinate is flipped by default, because GameMaker puts (0,0) at the top left and Blender at the bottom left.
- Colors are treated as sRGB and converted to and from Blender's linear color. You can turn this off in the options.
- On import, vertices at the same position are merged. Normals, UVs and colors stay per corner, so hard edges are kept.
- On export, all selected objects go into one triangle list. Normals, UVs and colors are only written if the mesh has them.
- If your faces show up inside out in the game, try *Flip Winding Order*.


## Limits

- Triangle lists, strips and fans are imported, plus lines and points. The built-in shapes  (block, cylinder, cone, ellipsoid, wall, floor) are skipped.
- Textures and materials are not imported or exported.
