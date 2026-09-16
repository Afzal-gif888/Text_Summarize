document.addEventListener('DOMContentLoaded', () => {
    const inputText = document.getElementById('inputText');
    const wordCount = document.getElementById('wordCount');
    const clearBtn = document.getElementById('clearBtn');
    const generateBtn = document.getElementById('generateBtn');
    const errorMsg = document.getElementById('errorMsg');
    
    const emptyState = document.getElementById('emptyState');
    const summaryResult = document.getElementById('summaryResult');
    const summaryText = document.getElementById('summaryText');
    const copyBtn = document.getElementById('copyBtn');
    
    const statOriginal = document.getElementById('statOriginal');
    const statSummary = document.getElementById('statSummary');
    const statReduction = document.getElementById('statReduction');

    const spinnerSvg = `
        <svg class="spinner" viewBox="0 0 50 50">
            <circle class="path" cx="25" cy="25" r="20" fill="none" stroke-width="5"></circle>
        </svg>
    `;
    const btnOriginalContent = `
        <span class="btn-icon">✦</span>
        <span class="btn-text-content">Generate Summary</span>
    `;

    // Update word count on input
    inputText.addEventListener('input', () => {
        const text = inputText.value.trim();
        const words = text ? text.split(/\s+/).length : 0;
        wordCount.textContent = `${words} word${words !== 1 ? 's' : ''}`;
        
        if (text.length > 15000) {
            errorMsg.textContent = 'Character limit reached (approx. 15,000 chars).';
        } else {
            errorMsg.textContent = '';
        }
    });

    // Clear input
    clearBtn.addEventListener('click', () => {
        inputText.value = '';
        inputText.dispatchEvent(new Event('input'));
        errorMsg.textContent = '';
        
        // Reset UI to empty state
        emptyState.classList.remove('hidden');
        summaryResult.classList.add('hidden');
    });

    // Generate Summary
    generateBtn.addEventListener('click', async () => {
        const text = inputText.value.trim();
        
        // Basic frontend validation
        if (!text) {
            errorMsg.textContent = 'Please paste some text to summarize.';
            return;
        }
        
        const words = text.split(/\s+/).length;
        if (words < 20) {
            errorMsg.textContent = 'Text is too short. Please provide at least 20 words.';
            return;
        }
        
        if (text.length > 15000) {
            errorMsg.textContent = 'Text is too long. Please limit to approximately 15,000 characters.';
            return;
        }

        // Set loading state
        generateBtn.disabled = true;
        generateBtn.innerHTML = `${spinnerSvg}<span class="btn-text-content">Generating summary...</span>`;
        errorMsg.textContent = '';
        
        try {
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                // Display summary safely (textContent escapes HTML automatically)
                summaryText.textContent = data.summary;
                
                // Update stats
                statOriginal.textContent = data.original_words;
                statSummary.textContent = data.summary_words;
                
                const reduction = Math.round((1 - (data.summary_words / data.original_words)) * 100);
                statReduction.textContent = `${reduction}%`;
                
                // Show result card
                emptyState.classList.add('hidden');
                summaryResult.classList.remove('hidden');
            } else {
                errorMsg.textContent = data.error || 'An unexpected error occurred.';
            }
        } catch (error) {
            errorMsg.textContent = 'Network error. Please check if the server is running.';
            console.error('Summarization Error:', error);
        } finally {
            // Reset button state
            generateBtn.disabled = false;
            generateBtn.innerHTML = btnOriginalContent;
        }
    });

    // Copy to clipboard
    copyBtn.addEventListener('click', async () => {
        const textToCopy = summaryText.textContent;
        
        try {
            await navigator.clipboard.writeText(textToCopy);
            const originalText = copyBtn.textContent;
            copyBtn.textContent = 'Copied!';
            copyBtn.style.backgroundColor = '#10b981';
            copyBtn.style.color = '#ffffff';
            
            setTimeout(() => {
                copyBtn.textContent = 'Copy';
                copyBtn.style.backgroundColor = '';
                copyBtn.style.color = '';
            }, 2000);
        } catch (err) {
            console.error('Failed to copy text: ', err);
        }
    });
});
