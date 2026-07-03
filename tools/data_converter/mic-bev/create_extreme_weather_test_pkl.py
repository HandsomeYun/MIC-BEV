#!/usr/bin/env python3
import argparse
import pickle
from collections import defaultdict
from pathlib import Path


DEFAULT_KEYWORDS = ("rain", "fog", "extreme")


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pkl(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def iter_search_strings(info):
    yield str(info.get("scene_token", ""))
    yield str(info.get("token", ""))
    yield str(info.get("map_path", ""))

    for cam_info in info.get("cams", {}).values():
        for key in ("data_path", "filename", "img_path"):
            value = cam_info.get(key)
            if value:
                yield str(value)

    for sweep in info.get("sweeps", []):
        value = sweep.get("data_path")
        if value:
            yield str(value)


def info_matches_keywords(info, keywords):
    text = " ".join(iter_search_strings(info)).lower()
    return any(keyword in text for keyword in keywords)


def filter_extreme_weather_infos(infos, keywords):
    scene_infos = defaultdict(list)
    matching_scenes = set()

    for info in infos:
        scene_token = info.get("scene_token", info.get("token"))
        scene_infos[scene_token].append(info)
        if info_matches_keywords(info, keywords):
            matching_scenes.add(scene_token)

    return [
        info
        for scene_token, scene_group in scene_infos.items()
        if scene_token in matching_scenes
        for info in scene_group
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an extreme-weather test PKL from an existing V2XSet test PKL."
    )
    parser.add_argument(
        "--input",
        default="/path/to/M2I/M2I_pkl/v2xset_infos_temporal_test.pkl",
        help="Existing generated test PKL.",
    )
    parser.add_argument(
        "--output",
        default="/path/to/M2I/M2I_pkl/v2xset_infos_temporal_extreme_test.pkl",
        help="Filtered extreme-weather test PKL to write.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=list(DEFAULT_KEYWORDS),
        help="Case-insensitive keywords used to select extreme-weather scenes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    keywords = tuple(keyword.lower() for keyword in args.keywords)

    data = load_pkl(input_path)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else data
    extreme_infos = filter_extreme_weather_infos(infos, keywords)

    if isinstance(data, dict):
        out = dict(data)
        out["infos"] = extreme_infos
        metadata = dict(out.get("metadata", {}))
        metadata["source_pkl"] = str(input_path)
        metadata["extreme_weather_keywords"] = list(keywords)
        out["metadata"] = metadata
    else:
        out = extreme_infos

    dump_pkl(out, output_path)
    matched_scenes = len({info.get("scene_token", info.get("token")) for info in extreme_infos})
    total_scenes = len({info.get("scene_token", info.get("token")) for info in infos})
    print(
        f"[INFO] Matched {matched_scenes}/{total_scenes} scenes and "
        f"{len(extreme_infos)}/{len(infos)} samples using keywords {keywords}"
    )
    print(f"[INFO] Wrote {output_path}")


if __name__ == "__main__":
    main()
