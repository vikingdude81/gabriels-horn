import json
from collections import defaultdict

# Load results
data = json.load(open('results/results.json'))

# Aggregate by condition
stats = defaultdict(lambda: {'rep4':[], 'rep8':[], 'degen':0, 'total':0})

for r in data: 
    c = r['condition']
    stats[c]['rep4'].append(r['rep_rate_4gram'])
    stats[c]['rep8'].append(r['rep_rate_8gram'])
    stats[c]['total'] += 1
    if r.get('degeneration_step') is not None: 
        stats[c]['degen'] += 1

print("="*70)
print("GABRIEL'S HORN EXPERIMENT - SUMMARY RESULTS")
print("="*70)
print(f"Total runs: {len(data)}")
print()

# Print summary table
print(f"{'Condition':<25} {'Avg Rep-4':>10} {'Avg Rep-8':>10} {'Degen/Total':>12}")
print("-"*60)

for c in sorted(stats.keys()):
    avg_r4 = sum(stats[c]['rep4'])/len(stats[c]['rep4'])
    avg_r8 = sum(stats[c]['rep8'])/len(stats[c]['rep8'])
    label = str(c).encode('ascii', 'ignore').decode()
    degen_str = f"{stats[c]['degen']}/{stats[c]['total']}"
    print(f"{label:<25} {avg_r4:>10.3f} {avg_r8:>10.3f} {degen_str:>12}")

print()
print("Legend:")
print("-" * 60)
print("Rep-4/8 = Fraction of repeated 4/8-grams (lower is better)")
print("Degen   = Prompts where entropy collapsed / total prompts")
print()

# Check semantic coherence scores if available
coherence_scores = [r.get('semantic_coherence') for r in data if r.get('semantic_coherence') is not None]
if coherence_scores:
    print(f"Semantic Coherence Scores Available: {len(coherence_scores)}")
    print(f"Mean Coherence: {sum(coherence_scores)/len(coherence_scores):.4f}")
else:
    print("Note: Semantic coherence scores not available (sentence-transformers not installed)")

# Print entropy statistics for each condition
print()
print("="*70)
print("ENTROPY STATISTICS BY CONDITION")
print("="*70)

for c in sorted(stats.keys()):
    cr = [r for r in data if r['condition'] == c]
    avg_adj_h = sum(sum(r['adjusted_entropies'])/len(r['adjusted_entropies']) for r in cr if r['adjusted_entropies']) / len(cr)
    final_avg = sum(r['adjusted_entropies'][-1] if r['adjusted_entropies'] else 0 for r in cr) / len(cr)
    label = str(c).encode('ascii', 'ignore').decode()
    print(f"{label:<25}: Mean Adj H={avg_adj_h:.3f}, Final Adj H={final_avg:.3f}")

print()