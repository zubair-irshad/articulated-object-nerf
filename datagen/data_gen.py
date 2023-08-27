import sapien
from data_utils import *
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Data generation for NeRF training.")

    parser.add_argument("--config", type=str, required=True, help="Path to configuration file.")
    parser.add_argument("--urdf_file", type=str, help="file path to the urdf file of sapien")
    parser.add_argument("--output_dir", type=str, help="path to save the generated images")
    parser.add_argument("--resolution", type=int, default=[512, 512], nargs='+', help="Image resolution, w h, default: w = 512, h = 512")
    
    args = parser.parse_args()

    # Load and parse the JSON configuration file
    with open(args.config, "r") as config_file:
        config_data = json.load(config_file)

    required_args = ["urdf_file", "output_dir"]
    missing_args = [arg for arg in required_args if arg not in config_data]
    if missing_args:
        raise ValueError(f"Required argument(s) {', '.join(missing_args)} not found in the JSON configuration")

    # Update the args namespace with loaded JSON data
    for key, value in config_data.items():
        setattr(args, key, value)

    return args
        
def main(args):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer(offscreen_only=True)
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    
    urdf_path = args.urdf_file
    asset = loader.load_kinematic(str(urdf_path))
    # asset = loader.load(urdf_path)
    assert asset, 'URDF not loaded.'


    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    near, far = 0.1, 100
    width, height = args.resolution
    # width, height = 640, 480
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )

    output_path = P(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)


    splits = ('train', 'test', 'val')
    print("generating images for training...")
    gen_articulated_object_nerf_s1(100, 4, 'train', camera, asset, scene, object_path=output_path)
    print("generating images for validation...")
    gen_articulated_object_nerf_s1(50, 4, 'test', camera, asset, scene, object_path=output_path)
    print("generating images for testing...")
    gen_articulated_object_nerf_s1(50, 4, 'val', camera, asset, scene, object_path=output_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)