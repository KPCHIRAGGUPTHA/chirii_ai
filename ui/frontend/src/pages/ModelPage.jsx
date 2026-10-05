import React, { useEffect, useState } from 'react';
import { fetchModels } from '../services/api';

export default function ModelPage() {
  const [models, setModels] = useState([]);

  useEffect(() => {
    fetchModels()
      .then((data) => setModels(data.models || []))
      .catch(console.error);
  }, []);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">🧠 Model Architecture & Checkpoint Health</h1>
        <p className="page-subtitle">
          Detailed technical specification and SHA-256 byte-integrity matrix for MiniGPT Model C (6.61M parameters).
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
        <div className="card">
          <h3 className="card-title">📐 Model C Architectural Specs</h3>
          <table className="custom-table">
            <tbody>
              <tr>
                <td><strong>Trainable Parameters</strong></td>
                <td><span className="code-badge">6,613,504</span> (6.61 Million)</td>
              </tr>
              <tr>
                <td><strong>Transformer Blocks (`n_layer`)</strong></td>
                <td><span className="code-badge">8 Layers</span></td>
              </tr>
              <tr>
                <td><strong>Attention Heads (`n_head`)</strong></td>
                <td><span className="code-badge">8 Heads</span> (32 dim / head)</td>
              </tr>
              <tr>
                <td><strong>Embedding Dimension (`n_embd`)</strong></td>
                <td><span className="code-badge">256</span></td>
              </tr>
              <tr>
                <td><strong>Vocabulary Size (`vocab_size`)</strong></td>
                <td><span className="code-badge">1,024</span> Subwords (BPE)</td>
              </tr>
              <tr>
                <td><strong>Context Length (`block_size`)</strong></td>
                <td><span className="code-badge">128</span> Tokens</td>
              </tr>
              <tr>
                <td><strong>Weight Tying</strong></td>
                <td>Enabled (`wte` tied with `lm_head`)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3 className="card-title">🔒 Checkpoint & Tokenizer Health Matrix</h3>
          <table className="custom-table">
            <thead>
              <tr>
                <th>Asset</th>
                <th>Status</th>
                <th>Verified SHA-256 Hash</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Base Model (Phase 5D)</strong></td>
                <td><span className="badge badge-purple">Verified</span></td>
                <td><span className="code-badge" style={{ fontSize: '0.75rem' }}>6f934bc3f2ca1cff6173896562cc215d...</span></td>
              </tr>
              <tr>
                <td><strong>SFT Model (Phase 6E)</strong></td>
                <td><span className="badge badge-green">Verified</span></td>
                <td><span className="code-badge" style={{ fontSize: '0.75rem' }}>14f9335aa66d7168500c10cebaa20db3...</span></td>
              </tr>
              <tr>
                <td><strong>BPE Tokenizer (1024)</strong></td>
                <td><span className="badge badge-green">Verified</span></td>
                <td><span className="code-badge" style={{ fontSize: '0.75rem' }}>6436b59303ef54cb91c6656f378f5a9b...</span></td>
              </tr>
              <tr>
                <td><strong>NaN / Inf Check</strong></td>
                <td><span className="badge badge-green">0 Tensors</span></td>
                <td><span className="code-badge">100 / 100 Tensors Finite</span></td>
              </tr>
              <tr>
                <td><strong>Reload Determinism</strong></td>
                <td><span className="badge badge-green">100% Bitwise</span></td>
                <td><span className="code-badge">Run A == Run B</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
