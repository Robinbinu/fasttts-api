let currentMode = 'http'; // 'http' or 'websocket'
let websocket = null;
let audioContext = null;
let audioQueue = [];
let isPlaying = false;
let typingTimer = null;
const typingDelay = 1000; // 1 second after user stops typing
let currentAudioChunks = [];
let isPlayingAudio = false;

async function setEngine() {
    var engine = document.getElementById("engine").value;
    await fetch('/set_engine?engine_name=' + engine);
}

async function speak() {
    if (currentMode === 'http') {
        await speakHTTP();
    } else {
        await speakWebSocket();
    }
}

async function speakHTTP() {
    var text = document.getElementById("text").value;
    try {
        var url = '/tts?text=' + encodeURIComponent(text);
        var audio = document.getElementById("audio");
        audio.src = url;
        audio.play();
    } catch (error) {
        console.error('Error during fetch or audio playback:', error);
    }
}

async function speakWebSocket() {
    var text = document.getElementById("text").value;
    if (!text.trim()) {
        console.log('No text to send');
        return;
    }

    try {
        console.log('Speaking via WebSocket:', text);
        updateStatus('Connecting to WebSocket...');
        
        // Initialize WebSocket if not connected
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            console.log('Initializing WebSocket connection');
            await initWebSocket();
        }

        // Clear previous audio chunks only if not playing
        if (!isPlayingAudio) {
            console.log('Clearing previous audio chunks');
            currentAudioChunks = [];
        }
        
        // Send text to server
        console.log('Sending text to server');
        websocket.send(text);
        updateStatus('Generating audio...');
        
        // Clear the text field after sending
        document.getElementById("text").value = '';
        
    } catch (error) {
        console.error('Error during WebSocket communication:', error);
        updateStatus('Error: ' + error.message);
    }
}

function initWebSocket() {
    return new Promise((resolve, reject) => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        websocket = new WebSocket(wsUrl);
        audioQueue = [];
        
        websocket.onopen = () => {
            console.log('WebSocket connected');
            updateStatus('WebSocket connected');
            resolve();
        };

        websocket.onmessage = async (event) => {
            console.log('WebSocket message received, type:', typeof event.data);
            
            try {
                // Parse JSON message
                const message = JSON.parse(event.data);
                console.log('Parsed message:', message);
                
                // Check for audio output
                if (message.audioOutput && message.audioOutput.audio) {
                    const isHeader = message.audioOutput.isHeader || false;
                    console.log('Received audio chunk (base64), isHeader:', isHeader);
                    
                    if (isHeader && message.audioOutput.sampleRate) {
                        console.log('WAV header with sample rate:', message.audioOutput.sampleRate);
                    }
                    
                    // Decode base64 audio to binary
                    const base64Audio = message.audioOutput.audio;
                    const binaryString = atob(base64Audio);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }
                    const arrayBuffer = bytes.buffer;
                    
                    console.log('Decoded audio chunk, size:', arrayBuffer.byteLength);
                    currentAudioChunks.push(arrayBuffer);
                }
                
                // Check for final output
                if (message.finalOutput && message.finalOutput.isFinal) {
                    console.log('Received final output signal');
                    updateStatus('Audio generation complete');
                    
                    // Play all accumulated chunks
                    if (currentAudioChunks.length > 0 && !isPlayingAudio) {
                        console.log('Playing all accumulated chunks');
                        playAudioFromBytes();
                    }
                }
            } catch (error) {
                console.error('Error processing WebSocket message:', error);
            }
        };

        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateStatus('WebSocket error');
            reject(error);
        };

        websocket.onclose = () => {
            console.log('WebSocket closed');
            updateStatus('WebSocket disconnected');
            websocket = null;
        };
    });
}

function playAudioFromBytes() {
    console.log('playAudioFromBytes called, chunks:', currentAudioChunks.length);
    if (currentAudioChunks.length === 0) {
        console.log('No audio chunks to play');
        return;
    }
    
    // Don't start a new playback if already playing
    if (isPlayingAudio) {
        console.log('Already playing audio, skipping');
        return;
    }

    // Simply concatenate all chunks (header + audio data already in WAV format)
    const audioBlob = new Blob(currentAudioChunks, { type: 'audio/wav' });
    console.log('Created audio blob, size:', audioBlob.size);
    const audioUrl = URL.createObjectURL(audioBlob);
    
    const audio = document.getElementById("audio");
    isPlayingAudio = true;
    audio.src = audioUrl;
    audio.play().then(() => {
        console.log('Audio playing successfully');
        updateStatus('Playing audio');
    }).catch(err => {
        console.error('Error playing audio:', err);
        updateStatus('Error playing audio');
        isPlayingAudio = false;
    });
    
    // When audio ends, mark as not playing
    audio.onended = () => {
        console.log('Audio finished playing');
        isPlayingAudio = false;
        updateStatus('Ready');
    };
    
    // Clear the chunks for next playback
    currentAudioChunks = [];
    console.log('Audio chunks cleared for next playback');
}

function updateStatus(message) {
    const statusEl = document.getElementById('status');
    if (statusEl) {
        statusEl.textContent = message;
    }
}

function setMode(mode) {
    currentMode = mode;
    
    // Update button styles
    const httpBtn = document.getElementById('httpMode');
    const wsBtn = document.getElementById('wsMode');
    const speakBtn = document.getElementById('speakButton');
    
    if (mode === 'http') {
        httpBtn.classList.add('active');
        wsBtn.classList.remove('active');
        updateStatus('Mode: HTTP');
        speakBtn.style.display = 'block';
        
        // Close WebSocket if open
        if (websocket) {
            websocket.close();
            websocket = null;
        }
    } else {
        wsBtn.classList.add('active');
        httpBtn.classList.remove('active');
        updateStatus('Mode: WebSocket (not connected)');
        speakBtn.style.display = 'none'; // Hide speak button in websocket mode
    }
}

function handleTextInput() {
    // Only auto-send in websocket mode
    if (currentMode !== 'websocket') {
        return;
    }
    
    // Clear the previous timer
    clearTimeout(typingTimer);
    
    // Set a new timer
    typingTimer = setTimeout(() => {
        const text = document.getElementById("text").value;
        if (text.trim()) {
            speakWebSocket();
        }
    }, typingDelay);
}

async function fetchVoices() {
    try {
        var response = await fetch('/voices');
        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.statusText);
        }
        var data = await response.json();
        var voicesDropdown = document.getElementById("voice");
        voicesDropdown.innerHTML = ''; // Clear previous options
        data.forEach(function(voice) {
            var option = document.createElement("option");
            option.text = voice;
            option.value = voice;
            voicesDropdown.add(option);
        });
    } catch (error) {
        console.error('Error fetching voices:', error);
    }
}

async function setVoice() {
    var voice = document.getElementById("voice").value;
    try {
        var response = await fetch('/setvoice?voice_name=' + encodeURIComponent(voice));
        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.statusText);
        }
        console.log('Voice set successfully:', voice);
    } catch (error) {
        console.error('Error setting voice:', error);
    }
}

document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById("text").value = "This is a text to speech demo text";
    document.getElementById("speakButton").addEventListener("click", speak);
    document.getElementById("engine").addEventListener("change", async function() {
        await setEngine();
        await fetchVoices();
    });
    document.getElementById("voice").addEventListener("change", setVoice);
    
    // Mode switcher buttons
    document.getElementById("httpMode").addEventListener("click", () => setMode('http'));
    document.getElementById("wsMode").addEventListener("click", () => setMode('websocket'));
    
    // Add input listener for auto-send in websocket mode
    const textArea = document.getElementById("text");
    textArea.addEventListener("input", handleTextInput);
    
    // Clear timer when user starts typing again
    textArea.addEventListener("keydown", () => {
        if (currentMode === 'websocket') {
            clearTimeout(typingTimer);
        }
    });

    fetchVoices();
});
