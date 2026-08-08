import json
from collections import defaultdict

# Load latest results
data = json.load(open('results/results.json'))

print("="*80)
print("GABRIEL'S HORN - EXPERIMENT RESULTS")
print("="*80)

stats = defaultdict(lambda: {'rep4':[], 'rep8':[], 'degen':0, 'total':0})

for r in data: 
    c = r['condition']
    stats[c]['rep4'].append(r['rep_rate_4gram'])
    stats[c]['rep8'].append(r['rep_rate_8gram'])
    stats[c]['total'] += 1
    if r.get('degeneration_step') is not None: 
        stats[c]['degen'] += 1

print(f"\nTotal runs: {len(data)}")
print()
print(f"{'Condition':<25} {'Avg Rep-4':>10} {'Avg Rep-8':>10} {'Degen/Total':>12}")
print("-"*60)

for c in sorted(stats.keys()):
    avg_r4 = sum(stats[c]['rep4'])/len(stats[c]['rep4'])
    avg_r8 = sum(stats[c]['rep8'])/len(stats[c]['rep8'])
    label = str(c).encode('ascii', 'ignore').decode()
    degen_str = f"{stats[c]['degen']}/{stats[c]['total']}"
    print(f"{label:<25} {avg_r4:>10.3f} {avg_r8:>10.3f} {degen_str:>12}")

print()
print("Legend: Rep-4/8 = repeated n-gram fraction (lower is better)")
print("        Degen   = prompts where entropy collapsed")