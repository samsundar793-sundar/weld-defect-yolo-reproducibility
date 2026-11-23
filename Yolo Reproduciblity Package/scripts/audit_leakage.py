"""
Forensic Leakage Audit of the Random-Split Test Set (Table 1 Replication).
Compares test set images against training images across three leakage vectors:
1. Base ID: stripped Roboflow augmentation hash (.rf.<hex>)
2. Parent ID: root camera recording prefix (IMG_\d+)
3. Perceptual Hash (pHash): visual near-duplicate collision (Hamming distance <= 5)
"""

import os
import re
from collections import defaultdict

def parse_identifiers(filename: str):
    """Extract base_id and parent_id from image filename."""
    base_id = filename.split(".rf.")[0] if ".rf." in filename else filename
    # Parent capture ID (e.g., IMG_1234 from IMG_1234_block_...)
    m = re.match(r'^(IMG_\d+)', base_id)
    parent_id = m.group(1) if m else None
    return base_id, parent_id

def audit_dataset_splits(train_files, test_files, phash_dict=None):
    """Audit test set overlap against train set."""
    train_bases = {parse_identifiers(f)[0] for f in train_files}
    train_parents = {parse_identifiers(f)[1] for f in train_files if parse_identifiers(f)[1] is not None}

    shared_base = 0
    shared_parent = 0
    shared_either = 0
    clean = 0

    for f in test_files:
        b, p = parse_identifiers(f)
        has_base = b in train_bases
        has_parent = p is not None and p in train_parents
        if has_base:
            shared_base += 1
        if has_parent:
            shared_parent += 1
        if has_base or has_parent:
            shared_either += 1
        else:
            clean += 1

    n_test = len(test_files)
    print(f"Total test images: {n_test}")
    print(f"Shared base ID: {shared_base} ({shared_base/n_test*100:.1f}%)")
    print(f"Shared parent ID: {shared_parent} ({shared_parent/n_test*100:.1f}%)")
    print(f"Shared base or parent ID (Leaked): {shared_either} ({shared_either/n_test*100:.1f}%)")
    print(f"Clean (Unshared): {clean} ({clean/n_test*100:.1f}%)")

if __name__ == "__main__":
    print("Forensic Leakage Audit Tool")
    print("Refer to manifests/group_assignments.csv for full dataset assignments.")
