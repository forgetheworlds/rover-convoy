"""Render the MuJoCo convoy with a follow camera and a measured slider readout."""

import argparse
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import DT, FIELD, SLIDER_LIMIT, SPAN, XML
from field_pair import OBS_SIZE, FieldPairEnv

WIDTH, HEIGHT = 1440, 810
OUTPUT = Path(__file__).resolve().parents[1] / 'runs' / 'render'


def visual_model():
    """Add zero-mass, non-contact geometry to the simulation XML for rendering."""
    xml = ET.parse(XML).getroot()
    visual = xml.find('visual')
    visual.find('global').attrib.update(offwidth=str(WIDTH), offheight=str(HEIGHT), fovy='37')
    visual.find('headlight').attrib.update(ambient='0.34 0.35 0.39', diffuse='0.44 0.45 0.50')
    ET.SubElement(visual, 'quality', shadowsize='4096', offsamples='4')
    ET.SubElement(visual, 'map', shadowclip='0.9', shadowscale='0.72', znear='0.025', zfar='12')
    ET.SubElement(xml, 'statistic', extent='24', center='12 0 0')

    assets = xml.find('asset')
    ET.SubElement(assets, 'texture', type='skybox', builtin='gradient',
                  rgb1='0.20 0.31 0.43', rgb2='0.72 0.74 0.73', width='512', height='1024')
    ET.SubElement(assets, 'texture', name='sand', type='2d', builtin='checker',
                  rgb1='0.53 0.47 0.39', rgb2='0.54 0.48 0.40',
                  mark='random', markrgb='0.40 0.37 0.31', random='0.12', width='256', height='256')
    ET.SubElement(assets, 'material', name='yard', texture='sand', texrepeat='2.5 2.5',
                  texuniform='true', specular='0.03', shininess='0.05')

    world = xml.find('worldbody')
    world.find('geom').attrib.update(material='yard', rgba='0.95 0.96 0.98 1')
    world.find('light').attrib.update(pos='14 -5 7', dir='-0.38 0.44 -1', castshadow='true',
                                     diffuse='0.75 0.69 0.57', specular='0.26 0.24 0.21')
    ET.SubElement(world, 'light', directional='true', dir='0.6 -0.1 -0.7',
                  diffuse='0.17 0.25 0.34', castshadow='false')

    bodies = {body.get('name'): body for body in world.iter('body')}
    axis_x = '0.70710678 0 0.70710678 0'
    axis_y = '0.70710678 0.70710678 0 0'
    black = '0.08 0.11 0.14 1'
    metal = '0.60 0.64 0.66 1'
    shell = '0.85 0.84 0.76 1'
    bronze = '0.70 0.40 0.18 1'

    def detail(body, shape, xyz, size, color, **attributes):
        ET.SubElement(bodies[body], 'geom', type=shape, pos=xyz, size=size, rgba=color,
                      mass='0', contype='0', conaffinity='0', **attributes)

    for rover, color in [('lead', '0.17 0.42 0.54 1'), ('follow', '0.29 0.40 0.43 1')]:
        bodies[rover].find('geom').set('rgba', color)
        detail(rover, 'box', '1.28 0 0.01', '0.20 0.58 0.115', black)
        detail(rover, 'box', '-1.28 0 0.01', '0.20 0.58 0.115', black)
        detail(rover, 'box', '0.94 0 0.42', '0.34 0.39 0.23', shell)
        detail(rover, 'box', '1.25 0 0.55', '0.035 0.34 0.115', black)
        detail(rover, 'box', '0.93 0 0.68', '0.40 0.43 0.055', color)
        detail(rover, 'cylinder', '0.94 0 0.83', '0.074 0.11', metal)
        detail(rover, 'cylinder', '0.94 0 0.963', '0.12 0.048', black)

        for y in (-1, 1):
            side = 'left' if y > 0 else 'right'
            detail(rover, 'box', f'-0.30 {0.55*y} 0.02', '0.70 0.040 0.12', black)
            detail(rover, 'box', f'-0.30 {0.595*y} 0.095', '0.54 0.010 0.028', color)
            detail(rover, 'box', f'0.98 {0.395*y} 0.53', '0.20 0.013 0.11', black)
            detail(rover, 'box', f'1.49 {0.38*y} 0.025', '0.012 0.106 0.043', '1 0.87 0.58 1')
            for axle in ('front', 'middle', 'rear'):
                wheel = f'{rover}_{axle}_{side}_wheel'
                detail(wheel, 'cylinder', f'0 {0.122*y} 0', '0.239 0.026', metal, quat=axis_y)
                detail(wheel, 'cylinder', f'0 {0.151*y} 0', '0.164 0.009', black, quat=axis_y)
                detail(wheel, 'cylinder', f'0 {0.171*y} 0', '0.075 0.028', metal, quat=axis_y)
                for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
                    detail(wheel, 'sphere',
                           f'{0.118*np.cos(angle)} {0.163*y} {0.118*np.sin(angle)}',
                           '0.020', metal)

    bodies['cargo'].find("geom[@name='payload']").set('rgba', shell)
    for x in (-4.76, -3.20, -0.94, 0.60):
        detail('cargo', 'cylinder', f'{x} 0 0.82', '0.695 0.055', bronze, quat=axis_x)
    for x in (-4.825, 0.675):
        detail('cargo', 'cylinder', f'{x} 0 0.82', '0.61 0.024', black, quat=axis_x)
        detail('cargo', 'cylinder', f'{x*1.004} 0 0.82', '0.38 0.035', metal, quat=axis_x)
    detail('front_turntable', 'cylinder', '0 0 0.015', '0.32 0.095', black)
    detail('cargo', 'box', '0 0 0.09', '0.20 0.31 0.10', metal)
    detail('rear_carriage', 'box', '0 0 0.06', '0.18 0.14 0.10', metal)
    return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))


def draw_path(scene, route, terrain_x, terrain_z, *, reverse=False):
    """Place render-only dashes along the actual path reference."""
    route_points = route.xy[::13].copy()
    if reverse:
        route_points[:, 0] = SPAN - route_points[:, 0]
        route_points[:, 1] *= -1
    empty = np.zeros(3)
    identity = np.eye(3).ravel()
    color = np.array([0.26, 0.84, 0.94, 1], dtype=np.float32)
    for start, finish in zip(route_points[:-1], route_points[1:]):
        start_xy = start + 0.12 * (finish - start)
        end_xy = start + 0.83 * (finish - start)
        a = np.r_[start_xy, np.interp(start_xy[0], terrain_x, terrain_z) + 0.021]
        b = np.r_[end_xy, np.interp(end_xy[0], terrain_x, terrain_z) + 0.021]
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, empty, empty, identity, color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.025, a, b)
        scene.ngeom += 1


def load_fonts():
    fonts = []
    system = Path('/System/Library/Fonts/Supplemental')
    for bold, size in [(True, 34), (False, 19), (True, 21)]:
        font = ImageFont.load_default(size)
        for face in (system / ('Arial Bold.ttf' if bold else 'Arial.ttf'),
                     'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'):
            try:
                font = ImageFont.truetype(face, size)
                break
            except OSError:
                continue
        fonts.append(font)
    return fonts


def annotate(rgb, env, fonts, controller_name, playback):
    frame = Image.fromarray(rgb)
    overlay = Image.new('RGBA', frame.size)
    draw = ImageDraw.Draw(overlay)
    title, text, values = fonts
    draw.rounded_rectangle((30, 26, 566, 131), radius=12, fill=(16, 26, 34, 230))
    draw.text((50, 41), 'Coupled rover transport', fill='#f7f7f3', font=title)
    label = f'{env.scenario.mass:g} kg payload  /  '
    label += f'Synthetic {env.scenario.slope_deg:g}° crest' if env.scenario.slope_deg else env.scenario.name.title()
    if env.speed_conditioned:
        label += f'  /  {env.speed_limit:g} m/s limit'
    draw.text((51, 91), label, fill='#c2d6dd', font=text)
    draw.rounded_rectangle((30, HEIGHT-94, WIDTH-30, HEIGHT-28), radius=12, fill=(16, 26, 34, 235))
    draw.text((51, HEIGHT-72), f'{controller_name}  /  {playback:g}x playback', fill='#d5e3e7', font=text)
    slide = float(env.metrics['slider'])
    draw.text((645, HEIGHT-74), f'Slider {100*slide:+.1f} cm', fill='#f7f7f3', font=values)
    speed = np.mean(env.metrics['body_speed'])
    draw.text((1000, HEIGHT-74), f'{speed:.2f} m/s  |  {max(0, float(env.metrics["progress"][1])):.1f} / {env.route.finish:.0f} m',
              fill='#f7f7f3', font=values)
    neutral = 914
    draw.line((neutral - 62, HEIGHT-59, neutral + 62, HEIGHT-59), fill='#64727d', width=3)
    draw.line((neutral, HEIGHT-64, neutral, HEIGHT-54), fill='#c2d6dd', width=2)
    indicator = neutral + 62 * np.clip(slide / SLIDER_LIMIT, -1, 1)
    draw.ellipse((indicator-5, HEIGHT-64, indicator+5, HEIGHT-54), fill='#63ddec')
    return Image.alpha_composite(frame.convert('RGBA'), overlay).convert('RGB')


def render(destination, policy_path, playback, scenario, reverse, seed, start, duration):
    from stable_baselines3 import PPO
    import torch

    torch.set_num_threads(1)
    policy = PPO.load(policy_path, device='cpu')
    env = FieldPairEnv(scenario, model=visual_model(), reverse=reverse,
                       direction_input=policy.observation_space.shape[0] > OBS_SIZE)
    base = env.env
    obs, _ = env.reset(seed=seed)
    terrain = base.model.hfield_data[:len(base.terrain_x)] * base.terrain_scale
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 124, -18, 11.6
    options = mujoco.MjvOption()
    options.sitegroup[:] = False
    fonts = load_fonts()
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('FFmpeg is required to export a replay.')
    direction = 'reverse' if reverse else 'forward'
    writer = subprocess.Popen([
        ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', f'{WIDTH}x{HEIGHT}', '-framerate',
        f'{playback / DT:g}', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(destination / f'{direction}.mp4'),
    ], stdin=subprocess.PIPE)
    frames = 0
    try:
        with mujoco.Renderer(base.model, height=HEIGHT, width=WIDTH) as renderer:
            while True:
                elapsed = base.ticks * DT
                if start <= elapsed < start + duration:
                    position = base.metrics['positions'].mean(axis=0)
                    if reverse:
                        position[:2] = [SPAN - position[0], -position[1]]
                    camera.lookat[:] = position + [0.75, 0.0, 0.73]
                    renderer.update_scene(base.data, camera=camera, scene_option=options)
                    draw_path(renderer.scene, base.route, base.terrain_x, terrain, reverse=reverse)
                    frame = annotate(renderer.render(), base, fonts,
                                     'Learned ' + direction, playback)
                    writer.stdin.write(frame.tobytes())
                    frames += 1
                action = policy.predict(obs, deterministic=True)[0]
                obs, _, terminal, timeout, info = env.step(action)
                if terminal or timeout:
                    break
        print(f'Completed {direction}: {info["success"]}; {base.ticks * DT:.1f} simulated seconds; '
              f'{info["stop_reason"] or "route completed"}.')
    finally:
        writer.stdin.close()
        status = writer.wait()
        env.close()
        if status or not frames:
            raise RuntimeError('The requested replay window could not be exported.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--course', choices=[s.name for s in FIELD], default=FIELD[0].name)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--seed', type=int, default=1071)
    parser.add_argument('--playback', type=float, default=2)
    parser.add_argument('--from-seconds', type=float, default=0)
    parser.add_argument('--clip-seconds', type=float, default=16)
    args = parser.parse_args()
    if args.playback <= 0 or args.from_seconds < 0 or args.clip_seconds <= 0:
        parser.error('Use a positive playback rate and clip duration, and a nonnegative start.')
    args.out.mkdir(parents=True, exist_ok=True)
    scenario = next(s for s in FIELD if s.name == args.course)
    render(args.out, args.model, args.playback, scenario, args.reverse, args.seed,
           args.from_seconds, args.clip_seconds)
