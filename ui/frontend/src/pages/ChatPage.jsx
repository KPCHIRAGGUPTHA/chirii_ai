import React, { useState } from 'react';
import { generateText } from '../services/api';

export default function ChatPage() {
  const [model, setModel] = useState('sft');
  const [prompt, setPrompt] = useState('Explain recursion in simple terms.');
  const [maxTokens, setMaxTokens] = useState(40);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);

  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await generateText({
        model,
        prompt,
        max_new_tokens: maxTokens,
        temperature,
        top_k: topK
      });
      setResponse(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const samplePrompts = [
    'What is the capital of France?',
    'Explain recursion in simple terms.',
    'Classify words as nouns or verbs: Work, Run, Book.',
    'Capitalize every word: learning artificial intelligence is exciting.'
  ];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">💬 Interactive Chat Playground</h1>
        <p className="page-subtitle">
          Chat directly with the CPU-hosted MiniGPT Model C (SFT Fine-Tuned or Base Pretrained model)
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem' }}>
        <div className="card">
          <div className="form-group">
            <label className="form-label">Model Selection</label>
            <select
              className="form-select"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="sft">Phase 6E SFT Model (Instruction Aligned — Default)</option>
              <option value="base">Phase 5D Base Model (Pretrained Baseline)</option>
            </select>
          </div>

          <div className="form-group">
            <div className="form-label">
              <span>Instruction / Prompt</span>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-subtle)' }}>Max context 128 tokens</span>
            </div>
            <textarea
              className="form-textarea"
              placeholder="Ask MiniGPT something..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
            />
          </div>

          <div style={{ marginBottom: '1.25rem' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem', fontWeight: 600 }}>
              Sample Prompts:
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {samplePrompts.map((p, idx) => (
                <button
                  key={idx}
                  className="btn btn-secondary"
                  style={{ fontSize: '0.8rem', padding: '0.4rem 0.75rem' }}
                  onClick={() => setPrompt(p)}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            onClick={handleGenerate}
            disabled={loading || !prompt.trim()}
          >
            {loading ? '⏳ Generating Response on CPU...' : '✨ Generate Response'}
          </button>

          {error && (
            <div style={{ marginTop: '1rem', color: 'var(--accent-danger)', fontSize: '0.9rem' }}>
              ⚠️ Error: {error}
            </div>
          )}

          {response && (
            <div style={{ marginTop: '1.5rem' }}>
              <h4 style={{ marginBottom: '0.5rem', fontSize: '0.95rem' }}>Generated Model Response:</h4>
              <div className="response-box">
                {response.response || <span style={{ color: 'var(--text-subtle)', italic: true }}>(Empty output or prompt-only)</span>}
              </div>

              <div className="response-meta">
                <div className="meta-item">
                  Model: <span className="meta-val">{response.model.toUpperCase()}</span>
                </div>
                <div className="meta-item">
                  Latency: <span className="meta-val">{response.generation_time_sec}s</span>
                </div>
                <div className="meta-item">
                  Tokens: <span className="meta-val">{response.tokens_generated}</span>
                </div>
                <div className="meta-item">
                  Speed: <span className="meta-val">{response.tokens_per_second} tok/s</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <h3 className="card-title">⚙ Generation Parameters</h3>

          <div className="form-group">
            <div className="form-label">
              <span>Temperature</span>
              <span className="slider-val">{temperature}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.5"
              step="0.05"
              className="form-slider"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
            />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>0.0 = Greedy / Deterministic, 0.8 = Balanced</span>
          </div>

          <div className="form-group">
            <div className="form-label">
              <span>Max New Tokens</span>
              <span className="slider-val">{maxTokens}</span>
            </div>
            <input
              type="range"
              min="10"
              max="128"
              step="5"
              className="form-slider"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
            />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>Upper limit constrained by block_size 128</span>
          </div>

          <div className="form-group">
            <div className="form-label">
              <span>Top-K Sampling</span>
              <span className="slider-val">{topK}</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              className="form-slider"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
            />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>0 = Disabled, 50 = Restrict to top 50 subwords</span>
          </div>

          <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            ℹ <strong>Inference Note:</strong> Executed strictly on CPU in <code>torch.no_grad()</code> evaluation mode. Prompt instruction tokens are automatically pre-formatted into standard Alpaca instruction structures.
          </div>
        </div>
      </div>
    </div>
  );
}
