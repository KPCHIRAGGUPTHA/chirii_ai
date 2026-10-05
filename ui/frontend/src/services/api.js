const API_BASE = '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Backend server unavailable');
  return res.json();
}

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch model info');
  return res.json();
}

export async function fetchPhase6Evaluation() {
  const res = await fetch(`${API_BASE}/evaluation/phase6`);
  if (!res.ok) throw new Error('Failed to fetch Phase 6 evaluation data');
  return res.json();
}

export async function generateText({ model = 'sft', prompt, max_new_tokens = 40, temperature = 0.8, top_k = 50, top_p = 0.9 }) {
  const res = await fetch(`${API_BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, prompt, max_new_tokens, temperature, top_k, top_p })
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Generation error' }));
    throw new Error(err.detail || 'Generation failed');
  }
  return res.json();
}

export async function compareModels({ prompt, max_new_tokens = 40, temperature = 0.8, top_k = 50, top_p = 0.9 }) {
  const res = await fetch(`${API_BASE}/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, max_new_tokens, temperature, top_k, top_p })
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Comparison error' }));
    throw new Error(err.detail || 'Model comparison failed');
  }
  return res.json();
}
