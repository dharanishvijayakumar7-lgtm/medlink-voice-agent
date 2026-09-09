import json

# Load curated medicines (high quality with dosages)
with open('indian_otc_medicines_curated.json', 'r', encoding='utf-8') as f:
    curated = json.load(f)

# Load original medicines
with open('indian_otc_medicines_final.json', 'r', encoding='utf-8') as f:
    original = json.load(f)

print(f"Curated medicines: {len(curated)}")
print(f"Original medicines: {len(original)}")

# Create merged list - curated first (they have better info), then original
merged = []
used_names = set()

# Add all curated medicines first
for med in curated:
    merged.append(med)
    used_names.add(med['name'].lower())

# Add original medicines that aren't already in curated
for med in original:
    if med['name'].lower() not in used_names:
        # Add dosage placeholder if missing
        if 'dosage' not in med:
            med['dosage'] = 'Refer to package instructions'
        if 'how_to_take' not in med:
            med['how_to_take'] = 'Take as directed on the package'
        if 'where_to_find' not in med:
            med['where_to_find'] = 'Any pharmacy or medical store'
        if 'disclaimer' not in med:
            med['disclaimer'] = 'This is general information, not medical advice.'
        
        merged.append(med)
        used_names.add(med['name'].lower())

print(f"Merged medicines: {len(merged)}")

# Save merged file
with open('indian_otc_medicines_merged.json', 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print("✓ Merged medicines saved to indian_otc_medicines_merged.json")
