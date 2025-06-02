import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

// Configure Transformers.js environment
env.allowLocalModels = false;
env.remoteURL = 'https://huggingface.co/';

class MnemogenomicsApp {
    constructor() {
        this.model = null;
        this.tokenizer = null;
        this.modelConfig = null;
        this.isGenerating = false;
        
        // Special tokens - will be loaded from tokenizer config
        this.specialTokens = {
            bos: '[BOS]',
            eos: '[EOS]',
            sep: '[SEP]',
            dna: '[DNA]',
            rna: '[RNA]',
            protein: '[PROTEIN]',
            desc: '[DESC]'
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.log('Mnemogenomics initialized. Ready to load a model.', 'info');
    }
    
    setupEventListeners() {
        // Model loading
        document.getElementById('loadExampleModelBtn').addEventListener('click', () => this.loadExampleModel());
        document.getElementById('loadFromUrlBtn').addEventListener('click', () => this.loadFromUrl());
        document.getElementById('applyCustomModelBtn').addEventListener('click', () => this.loadCustomFiles());
        
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
        
        // Input handling
        document.getElementById('uploadInputFile').addEventListener('change', (e) => this.handleFileUpload(e));
        document.getElementById('fetchInputFromUrlBtn').addEventListener('click', () => this.fetchInputFromUrl());
        
        // Example inputs
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.loadExampleInput(e.target.dataset.type));
        });
        
        // Generation mode change
        document.getElementById('generationMode').addEventListener('change', (e) => this.handleModeChange(e.target.value));
        
        // Parameter sliders
        document.getElementById('maxNewTokens').addEventListener('input', (e) => {
            document.getElementById('maxNewTokensValue').textContent = e.target.value;
        });
        document.getElementById('temperature').addEventListener('input', (e) => {
            document.getElementById('temperatureValue').textContent = e.target.value;
        });
        document.getElementById('topK').addEventListener('input', (e) => {
            document.getElementById('topKValue').textContent = e.target.value;
        });
        
        // Generate button
        document.getElementById('generateBtn').addEventListener('click', () => this.generate());
    }
    
    switchTab(tabId) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === tabId);
        });
    }
    
    async loadExampleModel() {
        this.updateModelStatus('Loading example model...', 'loading');
        this.log('Loading example model (Xenova/distilgpt2)...', 'info');
        
        try {
            // Use a small example model for demonstration
            const modelId = 'Xenova/distilgpt2';
            await this.loadModel(modelId);
            this.log('Example model loaded successfully!', 'success');
            this.updateModelStatus('Example model loaded', 'success');
        } catch (error) {
            this.log(`Failed to load example model: ${error.message}`, 'error');
            this.updateModelStatus('Failed to load example model', 'error');
        }
    }
    
    async loadFromUrl() {
        const urlInput = document.getElementById('modelUrlInput').value.trim();
        if (!urlInput) {
            this.log('Please enter a model URL or Hub ID', 'error');
            return;
        }
        
        this.updateModelStatus('Loading model from URL...', 'loading');
        this.log(`Loading model from: ${urlInput}`, 'info');
        
        try {
            await this.loadModel(urlInput);
            this.log('Model loaded successfully from URL!', 'success');
            this.updateModelStatus('Model loaded from URL', 'success');
        } catch (error) {
            this.log(`Failed to load model from URL: ${error.message}`, 'error');
            this.updateModelStatus('Failed to load model', 'error');
        }
    }
    
    async loadCustomFiles() {
        const configFile = document.getElementById('configFile').files[0];
        const tokenizerConfigFile = document.getElementById('tokenizerConfigFile').files[0];
        const vocabFile = document.getElementById('vocabFile').files[0];
        const modelFile = document.getElementById('modelFile').files[0];
        
        if (!configFile || !modelFile) {
            this.log('Please upload at least config.json and model files', 'error');
            return;
        }
        
        this.updateModelStatus('Loading custom model files...', 'loading');
        this.log('Loading custom model from uploaded files...', 'info');
        
        try {
            // Read config files
            const config = JSON.parse(await this.readFileAsText(configFile));
            this.modelConfig = config;
            
            if (tokenizerConfigFile) {
                const tokenizerConfig = JSON.parse(await this.readFileAsText(tokenizerConfigFile));
                this.updateSpecialTokens(tokenizerConfig);
            }
            
            if (vocabFile) {
                const vocab = JSON.parse(await this.readFileAsText(vocabFile));
                // Store vocab for tokenizer initialization
                this.vocab = vocab;
            }
            
            // Note: Loading custom model files directly in browser has limitations
            // This is a simplified implementation
            this.log('Custom file loading is limited in browser environment. Consider using URL loading instead.', 'warning');
            this.updateModelStatus('Custom files loaded (limited functionality)', 'success');
            
        } catch (error) {
            this.log(`Failed to load custom files: ${error.message}`, 'error');
            this.updateModelStatus('Failed to load custom files', 'error');
        }
    }
    
    async loadModel(modelId) {
        try {
            // Load the model using Transformers.js
            this.model = await pipeline('text-generation', modelId, {
                quantized: false // Use full precision for better quality
            });
            
            // Access tokenizer from the pipeline
            this.tokenizer = this.model.tokenizer;
            
            // Try to get model config
            if (this.model.model && this.model.model.config) {
                this.modelConfig = this.model.model.config;
                this.updateSpecialTokensFromModel();
            }
            
            return true;
        } catch (error) {
            throw new Error(`Model loading failed: ${error.message}`);
        }
    }
    
    updateSpecialTokens(tokenizerConfig) {
        if (tokenizerConfig.bos_token) this.specialTokens.bos = tokenizerConfig.bos_token;
        if (tokenizerConfig.eos_token) this.specialTokens.eos = tokenizerConfig.eos_token;
        if (tokenizerConfig.sep_token) this.specialTokens.sep = tokenizerConfig.sep_token;
        
        // Look for additional special tokens
        if (tokenizerConfig.additional_special_tokens) {
            tokenizerConfig.additional_special_tokens.forEach(token => {
                if (token.includes('DNA')) this.specialTokens.dna = token;
                else if (token.includes('RNA')) this.specialTokens.rna = token;
                else if (token.includes('PROTEIN')) this.specialTokens.protein = token;
                else if (token.includes('DESC')) this.specialTokens.desc = token;
            });
        }
    }
    
    updateSpecialTokensFromModel() {
        if (!this.tokenizer) return;
        
        const tokenizerConfig = this.tokenizer.config || {};
        this.updateSpecialTokens(tokenizerConfig);
    }
    
    handleModeChange(mode) {
        const targetTypeGroup = document.getElementById('targetSequenceTypeGroup');
        if (mode === 'desc_to_seq') {
            targetTypeGroup.classList.remove('hidden');
        } else {
            targetTypeGroup.classList.add('hidden');
        }
    }
    
    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        try {
            const content = await this.readFileAsText(file);
            document.getElementById('inputText').value = content;
            this.log(`Loaded file: ${file.name}`, 'info');
        } catch (error) {
            this.log(`Failed to read file: ${error.message}`, 'error');
        }
    }
    
    async fetchInputFromUrl() {
        const url = document.getElementById('fetchUrlInput').value.trim();
        if (!url) {
            this.log('Please enter a URL', 'error');
            return;
        }
        
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const content = await response.text();
            document.getElementById('inputText').value = content;
            this.log(`Fetched content from URL: ${url}`, 'info');
        } catch (error) {
            this.log(`Failed to fetch from URL: ${error.message}`, 'error');
        }
    }
    
    loadExampleInput(type) {
        const examples = {
            dna_seq: 'ATGGCTAGCTTAGCTAGTCCCTAGCTAGCTAGCTCGGACTAG',
            protein_seq: 'LASSQLASLMMASLASLAS',
            description: 'protein involved in cellular signaling and regulation of gene expression'
        };
        
        document.getElementById('inputText').value = examples[type] || '';
        this.log(`Loaded ${type.replace('_', ' ')} example`, 'info');
    }
    
    detectInputType(input) {
        const trimmed = input.trim();
        
        // Check if it's FASTA format
        if (trimmed.startsWith('>')) {
            // Parse FASTA to determine sequence type
            const lines = trimmed.split('\n');
            const sequence = lines.slice(1).join('').toUpperCase();
            
            if (/^[ATGC\s]+$/.test(sequence)) return 'dna_seq';
            if (/^[AUGC\s]+$/.test(sequence)) return 'rna_seq';
            if (/^[ACDEFGHIKLMNPQRSTVWY\s]+$/.test(sequence)) return 'protein_seq';
        }
        
        // If not FASTA, check if it looks like a raw sequence
        const upperInput = trimmed.toUpperCase();
        if (/^[ATGC]+$/.test(upperInput)) return 'dna_seq';
        if (/^[AUGC]+$/.test(upperInput)) return 'rna_seq';
        if (/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(upperInput)) return 'protein_seq';
        
        // Otherwise, assume it's a description
        return 'description';
    }
    
    parseFasta(input) {
        const records = [];
        const lines = input.trim().split('\n');
        let currentRecord = null;
        
        for (const line of lines) {
            if (line.startsWith('>')) {
                if (currentRecord) {
                    records.push(currentRecord);
                }
                currentRecord = {
                    header: line.substring(1),
                    sequence: ''
                };
            } else if (currentRecord) {
                currentRecord.sequence += line.trim();
            }
        }
        
        if (currentRecord) {
            records.push(currentRecord);
        }
        
        return records;
    }
    
    formatPrompt(input, mode, targetType = null) {
        let prompt = this.specialTokens.bos;
        
        if (mode === 'seq_to_desc') {
            // Sequence to description
            const inputType = this.detectInputType(input);
            const typeToken = this.specialTokens[inputType.replace('_seq', '')] || '';
            
            if (input.startsWith('>')) {
                // FASTA format - handle multiple records as few-shot
                const records = this.parseFasta(input);
                records.forEach((record, idx) => {
                    if (idx > 0) prompt += this.specialTokens.sep;
                    prompt += typeToken + record.sequence + this.specialTokens.desc;
                });
            } else {
                // Raw sequence
                prompt += typeToken + input.trim().toUpperCase() + this.specialTokens.desc;
            }
        } else {
            // Description to sequence
            const typeToken = this.specialTokens[targetType.toLowerCase()] || '[DNA]';
            prompt += this.specialTokens.desc + input.trim() + typeToken;
        }
        
        return prompt;
    }
    
    async generate() {
        if (!this.model) {
            this.log('Please load a model first', 'error');
            return;
        }
        
        if (this.isGenerating) {
            this.log('Generation already in progress', 'warning');
            return;
        }
        
        const input = document.getElementById('inputText').value.trim();
        if (!input) {
            this.log('Please enter input text', 'error');
            return;
        }
        
        this.isGenerating = true;
        document.getElementById('generateBtn').disabled = true;
        document.getElementById('outputText').value = 'Generating...';
        
        try {
            // Determine generation mode
            let mode = document.getElementById('generationMode').value;
            if (mode === 'auto') {
                const inputType = this.detectInputType(input);
                mode = inputType.includes('seq') ? 'seq_to_desc' : 'desc_to_seq';
                this.log(`Auto-detected mode: ${mode}`, 'info');
            }
            
            // Get target type for desc_to_seq
            const targetType = mode === 'desc_to_seq' ? 
                document.getElementById('targetSequenceType').value : null;
            
            // Format the prompt
            const prompt = this.formatPrompt(input, mode, targetType);
            this.log(`Prompt: ${prompt}`, 'info');
            
            // Get generation parameters
            const maxNewTokens = parseInt(document.getElementById('maxNewTokens').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            const topK = parseInt(document.getElementById('topK').value);
            
            // Generate
            const output = await this.model(prompt, {
                max_new_tokens: maxNewTokens,
                temperature: temperature,
                top_k: topK > 0 ? topK : undefined,
                do_sample: true,
                pad_token_id: this.tokenizer.pad_token_id,
                eos_token_id: this.tokenizer.eos_token_id
            });
            
            // Process output
            let generatedText = output[0].generated_text;
            
            // Remove the prompt from the output
            if (generatedText.startsWith(prompt)) {
                generatedText = generatedText.substring(prompt.length);
            }
            
            // Clean up the output
            generatedText = this.cleanOutput(generatedText, mode);
            
            // Format output appropriately
            if (mode === 'desc_to_seq') {
                generatedText = this.formatSequenceOutput(generatedText, targetType);
            }
            
            document.getElementById('outputText').value = generatedText;
            this.log('Generation completed successfully', 'success');
            
            // Show external tools for sequence output
            if (mode === 'desc_to_seq') {
                this.showExternalTools(generatedText, targetType);
            } else {
                this.hideExternalTools();
            }
            
        } catch (error) {
            this.log(`Generation failed: ${error.message}`, 'error');
            document.getElementById('outputText').value = 'Generation failed. Check console for details.';
        } finally {
            this.isGenerating = false;
            document.getElementById('generateBtn').disabled = false;
        }
    }
    
    cleanOutput(text, mode) {
        // Remove special tokens from output
        Object.values(this.specialTokens).forEach(token => {
            text = text.replace(new RegExp(token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), '');
        });
        
        // Clean up based on mode
        if (mode === 'desc_to_seq') {
            // Remove any non-sequence characters
            text = text.replace(/[^ATGCURNYKSWBDHV]/gi, '');
        }
        
        return text.trim();
    }
    
    formatSequenceOutput(sequence, type) {
        // Format as FASTA
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const header = `>generated_${type.toLowerCase()}_${timestamp}`;
        
        // Break sequence into 80-character lines
        const formattedSeq = sequence.match(/.{1,80}/g)?.join('\n') || sequence;
        
        return `${header}\n${formattedSeq}`;
    }
    
    showExternalTools(sequence, type) {
        const toolsSection = document.getElementById('externalToolsSection');
        toolsSection.classList.remove('hidden');
        
        // Extract just the sequence part (remove FASTA header)
        const cleanSeq = sequence.split('\n').slice(1).join('');
        
        const tools = {
            DNA: [
                { name: 'NCBI BLAST', url: `https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastSearch&PROG_DEF=blastn&BLAST_SPEC=&QUERY=${cleanSeq}` },
                { name: 'ExPASy Translate', url: `https://web.expasy.org/translate/` }
            ],
            RNA: [
                { name: 'RNAfold', url: `http://rna.tbi.univie.ac.at/cgi-bin/RNAWebSuite/RNAfold.cgi` },
                { name: 'NCBI BLAST', url: `https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastSearch&PROG_DEF=blastn&BLAST_SPEC=&QUERY=${cleanSeq}` }
            ],
            PROTEIN: [
                { name: 'NCBI BLAST', url: `https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastSearch&PROG_DEF=blastp&BLAST_SPEC=&QUERY=${cleanSeq}` },
                { name: 'UniProt', url: `https://www.uniprot.org/blast?query=${cleanSeq}` },
                { name: 'PDB Search', url: `https://www.rcsb.org/search` }
            ]
        };
        
        const toolLinks = tools[type] || [];
        toolsSection.innerHTML = '<h4>Analyze Sequence:</h4>' + 
            toolLinks.map(tool => 
                `<a href="${tool.url}" target="_blank" class="button-secondary small-btn">${tool.name}</a>`
            ).join('');
    }
    
    hideExternalTools() {
        document.getElementById('externalToolsSection').classList.add('hidden');
    }
    
    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = e => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }
    
    updateModelStatus(message, status) {
        const statusElement = document.getElementById('modelLoadStatus');
        statusElement.textContent = message;
        statusElement.className = status;
    }
    
    log(message, type = 'info') {
        const logArea = document.getElementById('logArea');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('p');
        logEntry.className = `log-${type}`;
        logEntry.textContent = `[${timestamp}] ${message}`;
        logArea.appendChild(logEntry);
        logArea.scrollTop = logArea.scrollHeight;
        
        // Also log to console
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MnemogenomicsApp();
});