from collections import defaultdict
import re

number_re = re.compile(r'(\d{1,3}(?:,\d{3})*|\d+)')
region_re = re.compile(r'^[^\d\n]+?(?=(?:\d|\bTOTAL\b|\bTOTALS?\b|$))', re.IGNORECASE)

def normalize_number(token: str) -> int:
    return int(token.replace(',', ''))

def parse_line(line: str):
    line = line.strip()
    if not line:
        return None, []
    m = region_re.search(line)
    region = m.group(0).strip() if m else "UNKNOWN"
    region = re.sub(r'\bTOTALS?\b', '', region, flags=re.IGNORECASE).strip()
    nums = []
    for tok in number_re.findall(line):
        try:
            nums.append(normalize_number(tok))
        except ValueError:
            continue
    return region if region else "UNKNOWN", nums

per_region = defaultdict(int)
with open("election.txt", "r", encoding="utf-8") as f:
    for raw in f:
        region, nums = parse_line(raw)
        if region and nums:
            per_region[region] += sum(nums)

print(per_region)

