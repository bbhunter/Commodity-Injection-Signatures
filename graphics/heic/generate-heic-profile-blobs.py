#!/usr/bin/env python3
"""
Copyright (c) 2020-2026 David H Hoyt LLC. All Rights Reserved.
"""
import argparse
import pathlib
import shutil
import struct
import subprocess
import tempfile


def run(command):
    subprocess.run(command, check=True)


def replace_once(data, old, new, description):
    count = data.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {description}, found {count}")
    return data.replace(old, new, 1)


def find_box(data, box_type):
    marker = box_type.encode("ascii")
    offsets = []
    start = 0
    while True:
        offset = data.find(marker, start)
        if offset < 0:
            break
        if offset >= 4:
            size = struct.unpack_from(">I", data, offset - 4)[0]
            if size >= 8 and offset - 4 + size <= len(data):
                offsets.append(offset - 4)
        start = offset + 1
    if len(offsets) != 1:
        raise RuntimeError(f"expected one {box_type} box, found {len(offsets)}")
    return offsets[0]


def box_size(data, offset):
    return struct.unpack_from(">I", data, offset)[0]


def set_box_size(data, offset, size):
    struct.pack_into(">I", data, offset, size)


def adjust_iloc_base_offsets(data, delta):
    iloc_offset = find_box(data, "iloc")
    version = data[iloc_offset + 8]
    if version != 0:
        raise RuntimeError(f"expected iloc version 0, found {version}")
    sizes_offset = iloc_offset + 12
    offset_size = data[sizes_offset] >> 4
    length_size = data[sizes_offset] & 0x0f
    base_offset_size = data[sizes_offset + 1] >> 4
    item_count = struct.unpack_from(">H", data, sizes_offset + 2)[0]
    position = sizes_offset + 4

    for _ in range(item_count):
        position += 4  # item_ID and data_reference_index
        base_offset = int.from_bytes(data[position:position + base_offset_size], "big")
        if base_offset != 0:
            adjusted_offset = base_offset + delta
            data[position:position + base_offset_size] = adjusted_offset.to_bytes(base_offset_size, "big")
        position += base_offset_size
        extent_count = struct.unpack_from(">H", data, position)[0]
        position += 2 + extent_count * (offset_size + length_size)


def add_colr_property(data, colr_box):
    result = bytearray(data)
    meta_offset = find_box(result, "meta")
    iprp_offset = find_box(result, "iprp")
    ipco_offset = find_box(result, "ipco")
    ipma_offset = find_box(result, "ipma")
    ipma_size = box_size(result, ipma_offset)
    association_count_offset = ipma_offset + 18
    associations_offset = association_count_offset + 1
    association_count = result[association_count_offset]
    if association_count != 5:
        raise RuntimeError(f"expected five item properties, found {association_count}")

    result[ipma_offset:ipma_offset] = colr_box
    ipma_offset += len(colr_box)
    association_count_offset += len(colr_box)
    associations_offset += len(colr_box)
    result[association_count_offset] = association_count + 1
    result[associations_offset + association_count:associations_offset + association_count] = bytes([6])
    set_box_size(result, ipma_offset, ipma_size + 1)
    set_box_size(result, ipco_offset, box_size(result, ipco_offset) + len(colr_box))
    set_box_size(result, iprp_offset, box_size(result, iprp_offset) + len(colr_box) + 1)
    set_box_size(result, meta_offset, box_size(result, meta_offset) + len(colr_box) + 1)
    adjust_iloc_base_offsets(result, len(colr_box) + 1)
    return bytes(result)


def replace_colr_property(data, colr_box):
    result = bytearray(data)
    meta_offset = find_box(result, "meta")
    iprp_offset = find_box(result, "iprp")
    ipco_offset = find_box(result, "ipco")
    colr_offset = find_box(result, "colr")
    old_size = box_size(result, colr_offset)
    delta = len(colr_box) - old_size
    result[colr_offset:colr_offset + old_size] = colr_box
    set_box_size(result, ipco_offset, box_size(result, ipco_offset) + delta)
    set_box_size(result, iprp_offset, box_size(result, iprp_offset) + delta)
    set_box_size(result, meta_offset, box_size(result, meta_offset) + delta)
    adjust_iloc_base_offsets(result, delta)
    return bytes(result)


def mutate_ipma_colr_index(data, index):
    result = bytearray(data)
    ipma_offset = find_box(result, "ipma")
    association_count_offset = ipma_offset + 18
    associations_offset = association_count_offset + 1
    if result[association_count_offset] != 5 or result[associations_offset + 1] != 2:
        raise RuntimeError("unexpected ipma association layout")
    result[associations_offset + 1] = index
    return bytes(result)


def truncate_hevc_sps_rbsp(data):
    result = bytearray(data)
    hvcc_offset = find_box(result, "hvcC")
    hvcc_end = hvcc_offset + box_size(result, hvcc_offset)
    array_count_offset = hvcc_offset + 30
    array_count = result[array_count_offset]
    offset = array_count_offset + 1
    mutated = 0

    for _ in range(array_count):
        nal_unit_type = result[offset] & 0x3f
        nal_count = struct.unpack_from(">H", result, offset + 1)[0]
        offset += 3
        for _ in range(nal_count):
            nal_size = struct.unpack_from(">H", result, offset)[0]
            offset += 2
            nal_end = offset + nal_size
            if nal_end > hvcc_end:
                raise RuntimeError("HEVC decoder configuration exceeds hvcC box")
            if nal_unit_type == 33:
                if nal_size < 4:
                    raise RuntimeError("HEVC SPS is too short to mutate")
                result[offset + 2:nal_end] = bytes(nal_size - 2)
                mutated += 1
            offset = nal_end

    if mutated != 1:
        raise RuntimeError(f"expected one HEVC SPS, found {mutated}")
    return bytes(result)


def set_meta_largesize(data, large_size):
    result = bytearray(data)
    meta_offset = find_box(result, "meta")
    result[meta_offset:meta_offset + 8] = struct.pack(">I4sQ", 1, b"meta", large_size)
    return bytes(result)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HEIF ICC QA carriers")
    parser.add_argument("valid_profile", type=pathlib.Path)
    parser.add_argument("malformed_profile", type=pathlib.Path)
    parser.add_argument("output_directory", type=pathlib.Path)
    return parser.parse_args()


def main():
    args = parse_args()
    valid_profile = args.valid_profile.read_bytes()
    malformed_profile = args.malformed_profile.read_bytes()
    if len(valid_profile) != len(malformed_profile):
        raise RuntimeError("valid and malformed profiles must have equal sizes")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="heif-icc-carriers.") as temporary_directory:
        temporary = pathlib.Path(temporary_directory)
        profiled_png = temporary / "profiled.png"
        unprofiled_png = temporary / "unprofiled.png"
        valid_heic = temporary / "valid-prof.heic"
        valid_avif = temporary / "valid-prof.avif"
        nclx_heif = temporary / "nclx-only.heif"

        run([
            "magick",
            "-size",
            "16x16",
            "gradient:red-blue",
            "-profile",
            str(args.valid_profile),
            f"PNG24:{profiled_png}",
        ])
        run(["magick", "-size", "16x16", "gradient:red-blue", f"PNG24:{unprofiled_png}"])
        run(["heif-enc", "--hevc", "-q", "80", "-o", str(valid_heic), str(profiled_png)])
        run(["heif-enc", "--avif", "-q", "80", "-o", str(valid_avif), str(profiled_png)])
        run(["heif-enc", "--hevc", "-q", "80", "-o", str(nclx_heif), str(unprofiled_png)])

        valid_heic_data = valid_heic.read_bytes()
        valid_avif_data = valid_avif.read_bytes()
        nclx_heif_data = nclx_heif.read_bytes()
        profile_marker = b"prof" + valid_profile

        if valid_heic_data.count(profile_marker) != 1:
            raise RuntimeError("HEIC encoder did not preserve the valid ICC profile exactly")
        if valid_avif_data.count(profile_marker) != 1:
            raise RuntimeError("AVIF encoder did not preserve the valid ICC profile exactly")

        shutil.copyfile(valid_heic, args.output_directory / "valid-prof.heic")
        shutil.copyfile(valid_avif, args.output_directory / "valid-prof.avif")
        shutil.copyfile(nclx_heif, args.output_directory / "nclx-only.heif")

        valid_ricc = replace_once(valid_heic_data, profile_marker, b"rICC" + valid_profile,
                                  "HEIC prof payload")
        malformed_prof = replace_once(valid_heic_data, profile_marker, b"prof" + malformed_profile,
                                      "HEIC prof payload")
        short_prof = replace_once(nclx_heif_data, b"nclx", b"prof", "HEIF nclx colour type")

        colr_offset = find_box(valid_heic_data, "colr")
        colr_size = box_size(valid_heic_data, colr_offset)
        prof_colr = valid_heic_data[colr_offset:colr_offset + colr_size]
        malformed_colr = replace_once(prof_colr, valid_profile, malformed_profile, "colr ICC payload")
        nclx_colr = struct.pack(">I4s4sHHHB", 19, b"colr", b"nclx", 1, 13, 6, 0x80)
        empty_prof_colr = struct.pack(">I4s4s", 12, b"colr", b"prof")
        empty_nclx_colr = struct.pack(">I4s4s", 12, b"colr", b"nclx")

        duplicate_prof = add_colr_property(valid_heic_data, prof_colr)
        conflicting_prof = add_colr_property(valid_heic_data, malformed_colr)
        conflicting_nclx = add_colr_property(valid_heic_data, nclx_colr)
        invalid_ipma = mutate_ipma_colr_index(valid_heic_data, 127)
        truncated_sps = truncate_hevc_sps_rbsp(valid_heic_data)
        empty_prof = replace_colr_property(valid_heic_data, empty_prof_colr)
        truncated_nclx = replace_colr_property(nclx_heif_data, empty_nclx_colr)
        overflow_meta = set_meta_largesize(valid_heic_data, 0x7fffffffffffffff)

        truncated_colr = bytearray(valid_heic_data)
        set_box_size(truncated_colr, colr_offset, 11)
        oversized_colr = bytearray(valid_heic_data)
        set_box_size(oversized_colr, colr_offset, 0xffffffff)

        (args.output_directory / "valid-rICC.heic").write_bytes(valid_ricc)
        (args.output_directory / "malformed-prof.heic").write_bytes(malformed_prof)
        (args.output_directory / "short-prof.hif").write_bytes(short_prof)
        (args.output_directory / "duplicate-prof.heic").write_bytes(duplicate_prof)
        (args.output_directory / "conflicting-prof.heic").write_bytes(conflicting_prof)
        (args.output_directory / "conflicting-nclx.heic").write_bytes(conflicting_nclx)
        (args.output_directory / "invalid-ipma-index.heic").write_bytes(invalid_ipma)
        (args.output_directory / "truncated-sps.heic").write_bytes(truncated_sps)
        (args.output_directory / "empty-prof.heic").write_bytes(empty_prof)
        (args.output_directory / "truncated-nclx.heic").write_bytes(truncated_nclx)
        (args.output_directory / "overflow-meta-largesize.heic").write_bytes(overflow_meta)
        (args.output_directory / "truncated-colr.heic").write_bytes(truncated_colr)
        (args.output_directory / "oversized-colr.heic").write_bytes(oversized_colr)


if __name__ == "__main__":
    main()
