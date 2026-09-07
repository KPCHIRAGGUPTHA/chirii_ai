document.addEventListener('DOMContentLoaded', () => {
    // Tab Navigation
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const target = btn.dataset.tab;
            document.getElementById(target).classList.add('active');
        });
    });

    // Range Sliders Value Display Sync
    const tempSlider = document.getElementById('tempSlider');
    const tempVal = document.getElementById('tempVal');
    tempSlider.addEventListener('input', (e) => tempVal.textContent = parseFloat(e.target.value).toFixed(2));

    const tokensSlider = document.getElementById('tokensSlider');
    const tokensVal = document.getElementById('tokensVal');
    tokensSlider.addEventListener('input', (e) => tokensVal.textContent = e.target.value);

    const topkSlider = document.getElementById('topkSlider');
    const topkVal = document.getElementById('topkVal');
    topkSlider.addEventListener('input', (e) => topkVal.textContent = e.target.value);

    // Preset Prompt Click Handlers
    const promptInput = document.getElementById('promptInput');
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            promptInput.value = btn.dataset.prompt;
        });
    });

    // Chat Output Handling
    const generateBtn = document.getElementById('generateBtn');
    const chatOutput = document.getElementById('chatOutput');
    const clearChatBtn = document.getElementById('clearChatBtn');

    clearChatBtn.addEventListener('click', () => {
        chatOutput.innerHTML = `
            <div class="system-message">
                👋 Welcome to <strong>Mini-GPT Studio</strong>! Output cleared.
            </div>
        `;
    });

    generateBtn.addEventListener('click', async () => {
        const prompt = promptInput.value;
        if (!prompt) return;

        // User bubble
        const userBubble = document.createElement('div');
        userBubble.className = 'chat-bubble user';
        userBubble.textContent = prompt;
        chatOutput.appendChild(userBubble);

        // AI bubble (Loading state)
        const aiBubble = document.createElement('div');
        aiBubble.className = 'chat-bubble ai';
        aiBubble.textContent = "⚡ Mini-GPT is thinking & sampling tokens...";
        chatOutput.appendChild(aiBubble);
        chatOutput.scrollTop = chatOutput.scrollHeight;

        generateBtn.disabled = true;

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    max_tokens: parseInt(tokensSlider.value),
                    temperature: parseFloat(tempSlider.value),
                    top_k: parseInt(topkSlider.value)
                })
            });

            const data = await res.json();
            if (data.success) {
                // Typewriter text animation for generated output
                aiBubble.textContent = "";
                const fullText = data.generated;
                let i = 0;
                const interval = setInterval(() => {
                    if (i < fullText.length) {
                        aiBubble.textContent += fullText.charAt(i);
                        i++;
                        chatOutput.scrollTop = chatOutput.scrollHeight;
                    } else {
                        clearInterval(interval);
                    }
                }, 15);
            } else {
                aiBubble.textContent = "⚠️ Error: " + (data.error || "Generation failed.");
            }
        } catch (err) {
            aiBubble.textContent = "⚠️ Server communication error: " + err.message;
        } finally {
            generateBtn.disabled = false;
        }
    });

    // Chart.js Setup for Real-time Loss Monitoring
    const ctx = document.getElementById('lossChart').getContext('2d');
    const lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Val Loss',
                    data: [],
                    borderColor: '#ec4899',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' },
                    title: { display: true, text: 'Step Iteration', color: '#9ca3af' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' },
                    title: { display: true, text: 'Cross-Entropy Loss', color: '#9ca3af' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f3f4f6' } }
            }
        }
    });

    // Training Form Submission
    const trainForm = document.getElementById('trainForm');
    const startTrainBtn = document.getElementById('startTrainBtn');
    let trainPollInterval = null;

    trainForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const reqBody = {
            iters: parseInt(document.getElementById('trainIters').value),
            batch_size: parseInt(document.getElementById('trainBatch').value),
            n_embd: parseInt(document.getElementById('trainEmbd').value),
            n_layer: parseInt(document.getElementById('trainLayers').value),
            custom_text: document.getElementById('customText').value
        };

        startTrainBtn.disabled = true;
        startTrainBtn.textContent = "⏳ Initializing Training...";

        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqBody)
            });
            const data = await res.json();
            if (data.success) {
                // Clear old chart data
                lossChart.data.labels = [];
                lossChart.data.datasets[0].data = [];
                lossChart.data.datasets[1].data = [];
                lossChart.update();

                // Start polling training status
                startPollingTrainStatus();
            } else {
                alert("Training launch error: " + data.error);
                startTrainBtn.disabled = false;
                startTrainBtn.textContent = "🚀 Start Training Run";
            }
        } catch (err) {
            alert("Connection error: " + err.message);
            startTrainBtn.disabled = false;
            startTrainBtn.textContent = "🚀 Start Training Run";
        }
    });

    function startPollingTrainStatus() {
        if (trainPollInterval) clearInterval(trainPollInterval);

        trainPollInterval = setInterval(async () => {
            try {
                const res = await fetch('/api/train/status');
                const status = await res.json();

                document.getElementById('trainStatusIndicator').textContent = "Status: " + status.status.toUpperCase();
                document.getElementById('metricStep').textContent = `${status.step} / ${status.max_iters}`;
                document.getElementById('metricTrainLoss').textContent = status.train_loss.toFixed(4);
                document.getElementById('metricValLoss').textContent = status.val_loss.toFixed(4);
                if (document.getElementById('metricPerplexity')) {
                    document.getElementById('metricPerplexity').textContent = status.val_perplexity ? status.val_perplexity.toFixed(2) : "0.00";
                }
                document.getElementById('metricTime').textContent = `${status.elapsed_sec}s`;

                // Update Loss Chart
                if (status.history && status.history.length > 0) {
                    lossChart.data.labels = status.history.map(h => h.step);
                    lossChart.data.datasets[0].data = status.history.map(h => h.train_loss);
                    lossChart.data.datasets[1].data = status.history.map(h => h.val_loss);
                    lossChart.update();
                }

                if (status.status === 'completed' || status.status === 'error') {
                    clearInterval(trainPollInterval);
                    startTrainBtn.disabled = false;
                    startTrainBtn.textContent = "🚀 Start Training Run";
                    fetchSystemInfo(); // Refresh system info parameter count
                }
            } catch (err) {
                console.error("Poll error:", err);
            }
        }, 1000);
    }

    // System Info Fetching
    async function fetchSystemInfo() {
        try {
            const res = await fetch('/api/info');
            const info = await res.json();

            if (info.device) {
                document.getElementById('deviceText').textContent = info.device + " Mode";
            }
            if (info.param_count) {
                const formatted = info.param_count > 1000000 
                    ? (info.param_count / 1000000).toFixed(2) + "M" 
                    : (info.param_count / 1000).toFixed(0) + "K";
                document.getElementById('paramCount').textContent = formatted;
            }
        } catch (err) {
            console.error("Info fetch error:", err);
        }
    }

    fetchSystemInfo();
});
