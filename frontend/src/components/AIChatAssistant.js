import React, { useState } from 'react';

const AIChatAssistant = ({ onSendMessage, messages, loading, activeCrop, currentStage, weather }) => {
  const [input, setInput] = useState('');

  const handleSend = (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    onSendMessage(input);
    setInput('');
  };

  const sampleQuestions = [
    `How much nitrogen should I apply for ${activeCrop || 'my crop'} during tillering?`,
    `What are early signs of stem borer in my field?`,
    `Is it safe to spray pesticides if rain is forecasted?`,
    `How to lower high soil pH naturally?`
  ];

  return (
    <div className="card" style={{ padding: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <h3 className="card-title">🤖 Generative AI Knowledge Assistant</h3>
          <p className="card-subtitle" style={{ marginBottom: 0 }}>
            Grounded in verified agronomy documents via <b>Retrieval-Augmented Generation (RAG)</b>.
          </p>
        </div>

        <div style={{ fontSize: '0.8rem', color: '#64748b', textAlign: 'right' }}>
          <div>Active Context: <b>{activeCrop || 'General Farm'}</b></div>
          <div>Stage: <b>{currentStage || 'Planning'}</b></div>
        </div>
      </div>

      {/* Suggested Quick Prompts */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem' }}>
        {sampleQuestions.map((q, idx) => (
          <button
            key={idx}
            onClick={() => onSendMessage(q)}
            style={{
              background: '#f1f5f9',
              border: '1px solid #cbd5e1',
              borderRadius: '20px',
              padding: '4px 12px',
              fontSize: '0.78rem',
              color: '#334155',
              cursor: 'pointer'
            }}
          >
            💬 {q}
          </button>
        ))}
      </div>

      {/* Chat Messages */}
      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((m, idx) => (
            <div key={idx} className={`chat-bubble ${m.role}`}>
              <div style={{ fontWeight: '700', fontSize: '0.75rem', marginBottom: '0.25rem', opacity: 0.8 }}>
                {m.role === 'user' ? '🧑 Farmer' : '🌱 AI Agronomist'}
              </div>
              <div style={{ whiteSpace: 'pre-line' }}>{m.content}</div>
              {m.sources && m.sources.length > 0 && (
                <div style={{ marginTop: '0.5rem', fontSize: '0.72rem', color: '#15803d' }}>
                  📚 <i>Verified Sources: {m.sources.join(', ')}</i>
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="chat-bubble assistant">
              <span style={{ fontStyle: 'italic', color: '#64748b' }}>Consulting agricultural knowledge base & synthesizing response...</span>
            </div>
          )}
        </div>

        <form onSubmit={handleSend} className="chat-input-bar">
          <input
            type="text"
            className="chat-input"
            placeholder="Ask anything about crop care, fertilizer dosage, disease symptoms, or weather impact..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button
            type="submit"
            className="btn-primary"
            style={{ width: 'auto', padding: '0.75rem 1.5rem' }}
            disabled={loading}
          >
            Ask Assistant
          </button>
        </form>
      </div>
    </div>
  );
};

export default AIChatAssistant;
