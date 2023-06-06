# wfc_python
An implementation of [mxgmn/WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse) in Python

WaveFunctionCollapse is an algorithm that generates bitmaps that are locally similar to the input bitmap. This is a translation into Python, based on the original implementation in C#.

usage;
Overlapping(
    image = image template, lots to use in samples
    N = tile size
    outputW = output width in pixels
    outputH = output height in pixels
    seamlessInput = if the template is seamless
    seamlessOutput = output image will be seamless
    symmetry = how many symmetries to generate from tiles
    )
    
run(
    fRate = save every frame
    refresh = update pixels every n frame
    limit = limit the amount frames to generate, 0 is no limit
    seed = random seed
    )
    
Tiled(
    name = select tileset from samples.xml
    subsetName = select subset from samples.xml if available, enter "default" for the default set or if there's not one available
    width = output width in tiles
    height = output height in tiles
    periodic = seamless
    black = black tiles for ungenerated background
    )
    
run() no options

Hopefully it's readable as is. If you have any suggestion to make it tidier & more understandable please leave a comment in issues.
    
