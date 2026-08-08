import json
from collections import defaultdict

data = json.load(open('results/results.json'))
stats = defaultdict(lambda: {'rep4':[], 'rep8':[], 'degen':0, 'total':0})

for r in data: 
    c = r['condition']
    stats[c]['rep4'].append(r['rep_rate_4gram'])
    stats[c]['rep8'].append(r['rep_rate_8gram'])
    stats[c]['total'] += 1
    if r['degeneration_step'] is not None: 
        stats[c]['degen'] += 1

print("=== NEW EXPERIMENT RESULTS ===")
for c in sorted(stats): 
    avg_r4 = sum(stats[c]['rep4'])/len(stats[c]['rep4'])
    avg_r8 = sum(stats[c]['rep8'])/len(stats[c]['rep8'])
    label = c.encode('ascii', 'ignore').decode()
    print(f"{label}: Rep-4={avg_r4:.3f}, Rep-8={avg_r8:.3f}, Degen={stats[c]['degen']}/{stats[c]['total']}")

# Check semantic coherence scores if available
print("\n=== SEMANTIC COHERENCE SCORES ===")
coherence_scores = [r['semantic_coherence'] for r in data if r.get('semantic_coherence') is not None]
if coherence_scores:
    print(f"Available coherence scores: {len(coherence_scores)}")
    print(f"Mean coherence: {sum(coherence_scores)/len(coherence_scores):.4f}")
else:
    print("No semantic coherence scores available (sentence-transformers may not be installed)")