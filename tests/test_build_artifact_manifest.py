from scripts.release.build_artifact_manifest import DEFAULT_PATTERNS, build_manifest


def test_build_manifest_hashes_matching_files(tmp_path):
    root = tmp_path
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    (root / "notes.txt").write_text("skip\n", encoding="utf-8")

    manifest = build_manifest(root, ["README.md"])

    assert manifest["file_count"] == 1
    assert manifest["files"][0]["path"] == "README.md"
    assert len(manifest["files"][0]["sha256"]) == 64


def test_default_patterns_include_final_external_inputs(tmp_path):
    root = tmp_path
    for name in [
        "release_metadata.json",
        "submission_metadata.json",
        "second_hardware_profile.json",
        "second_hardware_energy.json",
    ]:
        (root / name).write_text('{"ready": true}\n', encoding="utf-8")

    manifest = build_manifest(root, list(DEFAULT_PATTERNS))
    paths = {entry["path"] for entry in manifest["files"]}

    assert "release_metadata.json" in paths
    assert "submission_metadata.json" in paths
    assert "second_hardware_profile.json" in paths
    assert "second_hardware_energy.json" in paths
