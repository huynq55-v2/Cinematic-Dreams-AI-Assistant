const sendBtn = document.getElementById('send-btn');
const userInput = document.getElementById('user-input');
const chatBox = document.getElementById('chat-box');
const suggestionBox = document.getElementById('suggestion-box');

const relatedSuggestionsContainer = document.getElementById('related-suggestions-buttons');
const unrelatedSuggestionsContainer = document.getElementById('unrelated-suggestions-buttons');
const relatedSectionHeader = document.getElementById('related-suggestions-section');
const unrelatedSectionHeader = document.getElementById('unrelated-suggestions-section');

// ĐÃ SỬA: Sử dụng đường dẫn tương đối để dễ dàng deploy
const API_URL = '/ask';
const API_SUGGEST_URL = '/suggest-questions';

const showTypingIndicator = () => {
    const typingBubble = document.createElement('div');
    typingBubble.className = 'message bot';
    typingBubble.id = 'typing-indicator';
    typingBubble.innerHTML = `
        <div class="bubble">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>`;
    chatBox.appendChild(typingBubble);
    chatBox.scrollTop = chatBox.scrollHeight;
    return typingBubble;
};

const addMessage = (text, type, debugContext = null) => { 
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = text; 

    messageDiv.appendChild(bubble);

    if (type === 'bot' && debugContext) {
        const uniqueId = `debugCollapse-${Date.now()}-${Math.floor(Math.random() * 1000)}`;

        const debugToggleBtn = document.createElement('button');
        debugToggleBtn.className = 'debug-toggle-btn';
        debugToggleBtn.textContent = 'Show debug context (click me!)';
        debugToggleBtn.setAttribute('type', 'button');
        debugToggleBtn.setAttribute('data-bs-toggle', 'collapse');
        debugToggleBtn.setAttribute('data-bs-target', `#${uniqueId}`);
        debugToggleBtn.setAttribute('aria-expanded', 'false');
        debugToggleBtn.setAttribute('aria-controls', uniqueId);

        const debugInfoDiv = document.createElement('div');
        debugInfoDiv.className = 'collapse debug-info'; 
        debugInfoDiv.id = uniqueId;
        debugInfoDiv.textContent = debugContext; 

        bubble.appendChild(debugToggleBtn);
        bubble.appendChild(debugInfoDiv);   
    }

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
};

const handleSend = async (queryText = null) => {
    const query = queryText || userInput.value.trim();
    if (!query) return;

    addMessage(query, 'user');
    userInput.value = '';

    if (suggestionBox.style.display !== 'none') {
        suggestionBox.style.display = 'none'; 
    }

    const typingIndicator = showTypingIndicator();

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        
        if (!response.ok) throw new Error('Network response was not worthy.');
        
        const data = await response.json();
        chatBox.removeChild(typingIndicator);
        addMessage(data.answer, 'bot', data.debug_context); 
    } catch (error) {
        console.error("The Forge has cooled:", error);
        chatBox.removeChild(typingIndicator);
        addMessage("An error has occurred in the forge. The connection is lost.", 'bot', "Error: Check console for details."); 
    }
};

const loadSuggestions = async () => {
    try {
        const response = await fetch(API_SUGGEST_URL);
        const data = await response.json(); 

        relatedSuggestionsContainer.innerHTML = '';
        unrelatedSuggestionsContainer.innerHTML = '';

        const relatedSuggestions = data.related;
        if (relatedSuggestions && relatedSuggestions.length > 0) {
            relatedSuggestions.forEach(q => {
                const button = document.createElement('button');
                button.className = 'btn btn-info btn-sm'; 
                button.textContent = q;
                button.onclick = () => handleSend(q); 
                relatedSuggestionsContainer.appendChild(button);
            });
            relatedSectionHeader.style.display = 'block'; 
        } else {
            relatedSectionHeader.style.display = 'none'; 
        }

        const unrelatedSuggestions = data.unrelated;
        if (unrelatedSuggestions && unrelatedSuggestions.length > 0) {
            unrelatedSuggestions.forEach(q => {
                const button = document.createElement('button');
                button.className = 'btn btn-warning btn-sm'; 
                button.textContent = q;
                button.onclick = () => handleSend(q); 
                unrelatedSuggestionsContainer.appendChild(button);
            });
            unrelatedSectionHeader.style.display = 'block'; 
        } else {
            unrelatedSectionHeader.style.display = 'none'; 
        }
        
        if (relatedSuggestions.length > 0 || unrelatedSuggestions.length > 0) {
            suggestionBox.style.display = 'block';
        } else {
            suggestionBox.style.display = 'none';
        }

    } catch (error) {
        console.error("Failed to load suggestions:", error);
        suggestionBox.style.display = 'none';
    }
};

sendBtn.addEventListener('click', () => handleSend(null));
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault(); 
        handleSend(null);
    }
});

window.addEventListener('load', () => {
    addMessage("I am the AI Assistant of Cinematic Dreams. Ask me anything or try one of the suggestions below.", 'bot');
    loadSuggestions();
});
