from typing import Dict, List, Any, Optional
import argparse
import windio2cad.turbine as turbine
import solid
import subprocess


if __name__ == "__main__":

    # Create a command line parser
    parser = argparse.ArgumentParser(
        description="Translate a yaml definition of a semisubmersible platform into an OpenSCAD source file."
    )
    parser.add_argument("--input", help="Input .yaml file", required=True)
    parser.add_argument(
        "--output",
        help="Output .stl file. If the file exists it will be overwritten.",
        required=True,
    )
    parser.add_argument("--openscad", help="Path to OpenSCAD executable", required=True)
    parser.add_argument("--downsample",
                        default=1,
                        type=int,
                        help="Defaults to 1, meaning every cross section is rendered."
    )
    parser.add_argument(
        "--blade",
        default=None,
        nargs="?",
        const="blade"
    )

    parser.add_argument(
        "--tower",
        default=None,
        nargs="?",
        const="tower"
    )

    parser.add_argument(
        "--monopile",
        default=None,
        nargs="?",
        const="monopile"
    )

    parser.add_argument(
        "--floater",
        default=None,
        nargs="?",
        const="floater"
    )

    args = parser.parse_args()

    intermediate_openscad = "intermediate.scad"

    print(f"Input yaml: {args.input}")
    print(f"Output .stl: {args.output}")
    print(f"Blade downsampling: {args.downsample}")
    print(f"Intermediate OpenSCAD: {intermediate_openscad}")
    print(f"Path to OpenSCAD: {args.openscad}")
    print("Parsing .yaml ...")

    if args.blade == "blade":
        print("Rendering blade only...")
        blade = turbine.Blade(args.input)
        blade_object = blade.blade_hull(downsample_z=args.downsample)
        with open(intermediate_openscad, "w") as f:
            f.write("$fn = 25;\n")
            f.write(solid.scad_render(blade_object))
            
    elif args.tower == "tower":
        print("Rendering tower only...")
        tower = turbine.Tower(args.input)
        with open(intermediate_openscad, "w") as f:
            f.write("$fn = 25;\n")
            big_union = solid.union()(
                [tower.tower_union()]
            )
            f.write(solid.scad_render(big_union))
            
    elif args.monopile == "monopile":
        print("Rendering monopile only...")
        monopile = turbine.Tower(args.input, towerkey="monopile")
        with open(intermediate_openscad, "w") as f:
            f.write("$fn = 25;\n")
            big_union = solid.union()(
                [monopile.tower_union()]
            )
            f.write(solid.scad_render(big_union))
            
    elif args.floater == "floater":
        print("Rendering floater only...")
        fp = turbine.FloatingPlatform(args.input)
        with open(intermediate_openscad, "w") as f:
            f.write("$fn = 25;\n")
            big_union = solid.union()(
                [fp.members_union()]
            )
            f.write(solid.scad_render(big_union))

    else:
        print("Rendering everything...")
        blade = turbine.Blade(args.input)
        blade_object = blade.blade_hull(downsample_z=args.downsample)
        fp = turbine.FloatingPlatform(args.input)
        tower = turbine.Tower(args.input)
        monopile = turbine.Tower(args.input, towerkey="monopile")
        rna = turbine.RNA(args.input)
        with open(intermediate_openscad, "w") as f:
            f.write("$fn = 25;\n")
            big_union = solid.union()(
                [fp.members_union(), tower.tower_union(), monopile.tower_union(), rna.rna_union(blade_object)]
            )
            f.write(solid.scad_render(big_union))
            
    print("Creating .stl ...")
    subprocess.run([args.openscad, "-o", args.output, intermediate_openscad])
    print("Done!")
