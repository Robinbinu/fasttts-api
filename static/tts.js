// State
let currentMode = 'http';
let websocket = null;
let audioChunks = [];
let isPlayingAudio = false;
let typingTimer = null;
let currentAudio = null;
let audioSessionId = 0; // Track which audio session is current
const TYPING_DELAY = 1000;

// Stop any currently playing audio
function stopCurrentAudio() {
    // Increment session ID to invalidate any pending play attempts
    audioSessionId++;

    if (currentAudio) {
        // Remove event listeners to prevent them from firing during cleanup
        currentAudio.oncanplay = null;
        currentAudio.onended = null;
        currentAudio.onerror = null;

        // Pause if currently playing
        if (!currentAudio.paused) {
            currentAudio.pause();
        }

        // Clear the source to stop any loading
        currentAudio.removeAttribute('src');
        currentAudio.load(); // Reset the audio element

        currentAudio = null;
    }
    isPlayingAudio = false;
}

// HTTP Mode
async function speakHTTP() {
    const text = document.getElementById("text").value;
    if (!text.trim()) return;

    try {
        stopCurrentAudio();
        updateStatus('Loading audio...');

        const audio = document.getElementById("audio");
        currentAudio = audio;

        // Capture the session ID for this audio request
        const thisSessionId = audioSessionId;

        // Fetch the audio data first (for Safari/iOS compatibility)
        // This ensures we have complete audio before playing
        const response = await fetch('/tts?text=' + encodeURIComponent(text));

        if (!response.ok) {
            throw new Error('Failed to fetch audio: ' + response.statusText);
        }

        // Check if session is still valid
        if (thisSessionId !== audioSessionId) {
            console.log('Session expired during fetch');
            return;
        }

        // Get audio as blob
        const audioBlob = await response.blob();

        // Check session again after async operation
        if (thisSessionId !== audioSessionId) {
            console.log('Session expired after fetch');
            return;
        }

        // Create blob URL
        const audioUrl = URL.createObjectURL(audioBlob);

        // Set up event listeners
        audio.onended = () => {
            if (thisSessionId === audioSessionId) {
                updateStatus('Ready');
                isPlayingAudio = false;
            }
            URL.revokeObjectURL(audioUrl);
        };

        audio.onerror = (e) => {
            if (thisSessionId === audioSessionId) {
                console.error('Audio error:', e);
                updateStatus('Error loading audio');
                isPlayingAudio = false;
            }
            URL.revokeObjectURL(audioUrl);
        };

        // Set src with blob URL
        audio.src = audioUrl;
        audio.load();

        // CRITICAL for Safari/iOS: Call play() within the same event context
        // Now the audio is already loaded as a blob, so it will play immediately
        try {
            updateStatus('Playing...');
            const playPromise = audio.play();

            if (playPromise !== undefined) {
                playPromise
                    .then(() => {
                        if (thisSessionId === audioSessionId) {
                            console.log('HTTP audio playing');
                        }
                    })
                    .catch(error => {
                        if (thisSessionId !== audioSessionId) {
                            console.log('Play cancelled - session expired');
                            URL.revokeObjectURL(audioUrl);
                            return;
                        }

                        if (error.name === 'AbortError') {
                            console.log('Play interrupted - this is normal if a new request was made');
                        } else if (error.name === 'NotSupportedError') {
                            console.error('Audio format not supported:', error);
                            updateStatus('Audio format not supported on this browser');
                        } else {
                            console.error('Play error:', error);
                            updateStatus('Playback error: ' + error.message);
                        }
                        URL.revokeObjectURL(audioUrl);
                    });
            }
        } catch (error) {
            console.error('Error starting playback:', error);
            updateStatus('Error: ' + error.message);
            URL.revokeObjectURL(audioUrl);
        }

    } catch (error) {
        console.error('Error:', error);
        updateStatus('Error: ' + error.message);
    }
}

// WebSocket Mode
async function speakWebSocket() {
    const text = document.getElementById("text").value;
    if (!text.trim()) return;

    try {
        // Initialize WebSocket if needed
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            await initWebSocket();
        }

        // Send text to server
        websocket.send(text);
        updateStatus('Generating audio...');
        
        // Clear text field
        document.getElementById("text").value = '';
    } catch (error) {
        console.error('Error:', error);
        updateStatus('Error: ' + error.message);
    }
}

function initWebSocket() {
    return new Promise((resolve, reject) => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        websocket = new WebSocket(wsUrl);
        
        websocket.onopen = () => {
            console.log('WebSocket connected');
            updateStatus('Connected');
            resolve();
        };

        websocket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                
                // Handle audio chunks
                if (message.audioOutput?.audio) {
                    const audioData = base64ToArrayBuffer(message.audioOutput.audio);
                    audioChunks.push(audioData);
                }
                
                // Handle completion
                if (message.finalOutput?.isFinal) {
                    playAudio();
                }
            } catch (error) {
                console.error('Error processing message:', error);
            }
        };

        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateStatus('Connection error');
            reject(error);
        };

        websocket.onclose = () => {
            console.log('WebSocket closed');
            updateStatus('Disconnected');
            websocket = null;
        };
    });
}

function base64ToArrayBuffer(base64) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

function playAudio() {
    if (audioChunks.length === 0 || isPlayingAudio) return;

    stopCurrentAudio();
    isPlayingAudio = true;

    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);

    const audio = document.getElementById("audio");
    currentAudio = audio;

    // Capture the session ID for this audio request
    const thisSessionId = audioSessionId;

    // Set up event listeners
    audio.onended = () => {
        if (thisSessionId === audioSessionId) {
            isPlayingAudio = false;
            updateStatus('Ready');
        }
        URL.revokeObjectURL(audioUrl);
    };

    audio.onerror = (e) => {
        if (thisSessionId === audioSessionId) {
            console.error('Audio error:', e);
            updateStatus('Error loading audio');
            isPlayingAudio = false;
        }
        URL.revokeObjectURL(audioUrl);
    };

    // Set src and load
    audio.src = audioUrl;
    audio.load();

    // Clear chunks now
    audioChunks = [];

    // CRITICAL for Safari/iOS: Call play() immediately, not in oncanplay
    try {
        updateStatus('Playing...');
        const playPromise = audio.play();

        if (playPromise !== undefined) {
            playPromise
                .then(() => {
                    if (thisSessionId === audioSessionId) {
                        console.log('WebSocket audio playing');
                    }
                })
                .catch(error => {
                    if (thisSessionId !== audioSessionId) {
                        console.log('Play cancelled - session expired');
                        URL.revokeObjectURL(audioUrl);
                        return;
                    }

                    if (error.name === 'AbortError') {
                        console.log('Play interrupted - this is normal if a new request was made');
                    } else if (error.name === 'NotSupportedError') {
                        console.error('Audio format not supported:', error);
                        updateStatus('Audio format not supported on this browser');
                    } else {
                        console.error('Play error:', error);
                        updateStatus('Playback error: ' + error.message);
                    }
                    URL.revokeObjectURL(audioUrl);
                });
        }
    } catch (error) {
        console.error('Error starting playback:', error);
        updateStatus('Error: ' + error.message);
        URL.revokeObjectURL(audioUrl);
    }
}

// UI Functions
function updateStatus(message) {
    const statusEl = document.getElementById('status');
    if (statusEl) {
        statusEl.textContent = message;
    }
}

function setMode(mode) {
    currentMode = mode;
    const httpBtn = document.getElementById('httpMode');
    const wsBtn = document.getElementById('wsMode');
    const speakBtn = document.getElementById('speakButton');
    
    if (mode === 'http') {
        httpBtn.classList.add('active');
        wsBtn.classList.remove('active');
        speakBtn.style.display = 'block';
        updateStatus('Mode: HTTP');
        
        if (websocket) {
            websocket.close();
            websocket = null;
        }
    } else {
        wsBtn.classList.add('active');
        httpBtn.classList.remove('active');
        speakBtn.style.display = 'none';
        updateStatus('Mode: WebSocket');
    }
}

function handleTextInput() {
    if (currentMode !== 'websocket') return;
    
    clearTimeout(typingTimer);
    typingTimer = setTimeout(() => {
        const text = document.getElementById("text").value;
        if (text.trim()) {
            speakWebSocket();
        }
    }, TYPING_DELAY);
}

async function setEngine() {
    const engine = document.getElementById("engine").value;
    await fetch('/set_engine?engine_name=' + engine);
}

async function fetchVoices() {
    try {
        const response = await fetch('/voices');
        if (!response.ok) throw new Error('Failed to fetch voices');
        
        const voices = await response.json();
        const dropdown = document.getElementById("voice");
        dropdown.innerHTML = '';
        
        voices.forEach(voice => {
            const option = document.createElement("option");
            option.text = voice;
            option.value = voice;
            dropdown.add(option);
        });
    } catch (error) {
        console.error('Error fetching voices:', error);
    }
}

async function setVoice() {
    const voice = document.getElementById("voice").value;
    try {
        const response = await fetch('/setvoice?voice_name=' + encodeURIComponent(voice));
        if (!response.ok) throw new Error('Failed to set voice');
        console.log('Voice set:', voice);
    } catch (error) {
        console.error('Error setting voice:', error);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    const textArea = document.getElementById("text");
    textArea.value = "This is a text to speech demo text";
    
    // Event listeners
    document.getElementById("speakButton").addEventListener("click", () => {
        if (currentMode === 'http') speakHTTP();
        else speakWebSocket();
    });
    
    document.getElementById("engine").addEventListener("change", async () => {
        await setEngine();
        await fetchVoices();
    });
    
    document.getElementById("voice").addEventListener("change", setVoice);
    document.getElementById("httpMode").addEventListener("click", () => setMode('http'));
    document.getElementById("wsMode").addEventListener("click", () => setMode('websocket'));
    
    textArea.addEventListener("input", handleTextInput);
    textArea.addEventListener("keydown", () => {
        if (currentMode === 'websocket') clearTimeout(typingTimer);
    });

    fetchVoices();
});
