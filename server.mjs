import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import fetch from 'node-fetch';

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));
app.use(express.static('.'));

app.post('/api/openai/chat', async (req, res) => {
  try {
    const {
      model = 'gpt-4o-mini',
      system = 'You are helpful.',
      user = '',
      temperature = 0.7,
      history = []
    } = req.body || {};

    const messages = [{ role: 'system', content: system }];
    for (const m of history) {
      if (m.role === 'user' || m.role === 'assistant') messages.push(m);
    }
    if (user) messages.push({ role: 'user', content: user });

    const r = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model,
        messages,
        temperature,
        max_tokens: 400
      })
    });

    const j = await r.json();
    if (!r.ok) return res.status(r.status).json(j);
    const text = j.choices?.[0]?.message?.content || '';
    res.json({ text, raw: j });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post('/api/openai/image', async (req, res) => {
  try {
    const { prompt = '' } = req.body || {};
    const r = await fetch('https://api.openai.com/v1/images/generations', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ prompt, n: 1, size: '1024x1024' })
    });
    const j = await r.json();
    if (!r.ok) return res.status(r.status).json(j);
    const url = j.data?.[0]?.url;
    res.json({ url, raw: j });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post('/api/anthropic/chat', async (req, res) => {
  try {
    const {
      model = 'claude-3-7-sonnet-20250219',
      system = 'You are helpful.',
      user = '',
      temperature = 0.7
    } = req.body || {};

    const r = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json'
      },
      body: JSON.stringify({
        model,
        system,
        max_tokens: 400,
        temperature,
        messages: [{ role: 'user', content: user }]
      })
    });

    const j = await r.json();
    if (!r.ok) return res.status(r.status).json(j);
    const text = j.content?.[0]?.text || '';
    res.json({ text, raw: j });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

const PORT = process.env.PORT || 5173;
app.listen(PORT, () => {
  console.log(`Proxy + static server on http://localhost:${PORT}`);
});