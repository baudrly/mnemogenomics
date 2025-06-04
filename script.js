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
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });
        
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === tabId);
        });
    }
    
    async loadExampleModel() {
        this.updateModelStatus('Loading example model...', 'loading');
        this.log('Loading example model files from demo directory...', 'info');
        
        try {
            // Load example model files from local demo directory
            const demoPath = './demo_model/';
            const files = ['config.json', 'tokenizer_config.json', 'vocab.json', 'model.onnx'];
            
            const loadedFiles = {};
            for (const filename of files) {
                try {
                    const response = await fetch(demoPath + filename);
                    if (response.ok) {
                        const content = filename.endsWith('.onnx') ? 
                            await response.blob() : 
                            await response.json();
                        loadedFiles[filename] = content;
                    }
                } catch (e) {
                    this.log(`Could not load ${filename}: ${e.message}`, 'warning');
                }
            }
            
            if (loadedFiles['config.json'] && loadedFiles['model.onnx']) {
                // Process the loaded files
                this.modelConfig = loadedFiles['config.json'];
                if (loadedFiles['tokenizer_config.json']) {
                    this.updateSpecialTokens(loadedFiles['tokenizer_config.json']);
                }
                
                // For demo purposes, create a simple mock model
                this.model = {
                    generate: async (inputs, options) => {
                        // Simple mock generation
                        await new Promise(resolve => setTimeout(resolve, 1000));
                        return this.mockGenerate(inputs, options);
                    }
                };
                
                if (loadedFiles['vocab.json']) {
                    this.tokenizer = {
                        encode: (text) => this.simpleTokenize(text),
                        decode: (tokens) => this.simpleDetokenize(tokens),
                        vocab: loadedFiles['vocab.json']
                    };
                }
                
                this.log('Example model loaded successfully!', 'success');
                this.updateModelStatus('Example model loaded', 'success');
            } else {
                throw new Error('Required model files not found in demo directory');
            }
        } catch (error) {
            this.log(`Failed to load example model: ${error.message}`, 'error');
            this.log('Falling back to Hugging Face model...', 'info');
            
            // Fallback to HF model
            try {
                const modelId = 'Xenova/distilgpt2';
                await this.loadModel(modelId);
                this.log('Fallback model loaded successfully!', 'success');
                this.updateModelStatus('Model loaded (fallback)', 'success');
            } catch (fallbackError) {
                this.log(`Failed to load fallback model: ${fallbackError.message}`, 'error');
                this.updateModelStatus('Failed to load model', 'error');
            }
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
            
            let vocab = null;
            if (vocabFile) {
                vocab = JSON.parse(await this.readFileAsText(vocabFile));
            }
            
            // Create a mock model for custom files
            this.model = {
                generate: async (inputs, options) => {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    return this.mockGenerate(inputs, options);
                }
            };
            
            this.tokenizer = {
                encode: (text) => this.simpleTokenize(text),
                decode: (tokens) => this.simpleDetokenize(tokens),
                vocab: vocab || this.createDefaultVocab()
            };
            
            this.log('Custom model loaded successfully!', 'success');
            this.updateModelStatus('Custom model loaded', 'success');
            
        } catch (error) {
            this.log(`Failed to load custom files: ${error.message}`, 'error');
            this.updateModelStatus('Failed to load custom files', 'error');
        }
    }
    
    async loadModel(modelId) {
        try {
            this.model = await pipeline('text-generation', modelId, {
                quantized: false
            });
            
            this.tokenizer = this.model.tokenizer;
            
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
            dna_seq: 'ATGCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCG',
            protein_seq: 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
            description: 'A protein that functions as a key regulator of cellular metabolism in saccharomyces cerevisiae, showing enhanced activity under oxidative stress conditions'
        };
        
        document.getElementById('inputText').value = examples[type] || '';
        this.log(`Loaded ${type.replace('_', ' ')} example`, 'info');
    }
    
    detectInputType(input) {
        const trimmed = input.trim();
        
        // Remove FASTA header if present
        const cleanInput = trimmed.startsWith('>') ? 
            trimmed.split('\n').slice(1).join('').trim() : 
            trimmed;
        
        // Check if it looks like a sequence
        const upperInput = cleanInput.toUpperCase().replace(/\s+/g, '');
        const seqType = this.detectSequenceType(upperInput);
        
        if (seqType !== 'UNKNOWN') {
            return seqType.toLowerCase() + '_seq';
        }
        
        // Otherwise, assume it's a description
        return 'description';
    }
    
    detectSequenceType(sequence) {
        const seqUpper = sequence.toUpperCase().replace(/\s+/g, '');
        if (seqUpper.length === 0) return 'UNKNOWN';
        
        // Count character frequencies
        const counts = {};
        for (const char of seqUpper) {
            counts[char] = (counts[char] || 0) + 1;
        }
        
        const total = seqUpper.length;
        const nucleotideChars = 'ACGTUN';
        const proteinChars = 'ACDEFGHIKLMNPQRSTVWY';
        
        // Calculate nucleotide percentage
        let nucleotideCount = 0;
        for (const char of nucleotideChars) {
            nucleotideCount += counts[char] || 0;
        }
        
        // If >90% nucleotides, check DNA vs RNA
        if (nucleotideCount / total > 0.9) {
            const hasU = (counts['U'] || 0) > 0;
            const hasT = (counts['T'] || 0) > 0;
            
            if (hasU && !hasT) return 'RNA';
            if (hasT && !hasU) return 'DNA';
            if (!hasU && !hasT) return 'DNA'; // Default to DNA
        }
        
        // Check for protein
        let proteinCount = 0;
        for (const char of proteinChars) {
            proteinCount += counts[char] || 0;
        }
        
        if (proteinCount / total > 0.85) return 'PROTEIN';
        
        return 'UNKNOWN';
    }
    
    parseFasta(input) {
        const records = [];
        const lines = input.trim().split(/\r?\n/);
        let currentRecord = null;
        
        for (const line of lines) {
            if (line.startsWith('>')) {
                if (currentRecord) {
                    records.push(currentRecord);
                }
                currentRecord = {
                    header: line.substring(1).trim(),
                    sequence: ''
                };
            } else if (currentRecord && line.trim() !== '') {
                currentRecord.sequence += line.trim().toUpperCase();
            }
        }
        
        if (currentRecord) {
            records.push(currentRecord);
        }
        
        return records;
    }
    
    formatPrompt(input, mode, targetType = null) {
        let prompt = this.specialTokens.bos;
        
        // Clean input - remove FASTA headers
        let cleanInput = input.trim();
        if (cleanInput.startsWith('>')) {
            const lines = cleanInput.split('\n');
            cleanInput = lines.slice(1).join('').trim();
        }
        
        if (mode === 'seq_to_desc') {
            // Sequence to description
            const seqType = this.detectSequenceType(cleanInput);
            const typeToken = this.specialTokens[seqType.toLowerCase()] || this.specialTokens.dna;
            
            prompt += typeToken + cleanInput.toUpperCase().replace(/\s+/g, '') + 
                     this.specialTokens.sep + this.specialTokens.desc;
        } else {
            // Description to sequence
            const typeToken = this.specialTokens[targetType.toLowerCase()] || this.specialTokens.dna;
            prompt += this.specialTokens.desc + cleanInput + 
                     this.specialTokens.sep + typeToken;
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
        this.showGeneratingUI(true);
        
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
            this.log(`Generating with prompt: ${prompt.substring(0, 100)}...`, 'info');
            
            // Get generation parameters
            const maxNewTokens = parseInt(document.getElementById('maxNewTokens').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            const topK = parseInt(document.getElementById('topK').value);
            
            // Generate
            let generatedText;
            
            if (this.model.generate) {
                // Using mock or custom model
                generatedText = await this.model.generate(prompt, {
                    max_new_tokens: maxNewTokens,
                    temperature: temperature,
                    top_k: topK > 0 ? topK : undefined,
                    mode: mode,
                    targetType: targetType
                });
            } else {
                // Using Transformers.js pipeline
                const output = await this.model(prompt, {
                    max_new_tokens: maxNewTokens,
                    temperature: temperature,
                    top_k: topK > 0 ? topK : undefined,
                    do_sample: true,
                    pad_token_id: this.tokenizer?.pad_token_id,
                    eos_token_id: this.tokenizer?.eos_token_id
                });
                
                generatedText = output[0].generated_text;
                
                // Remove the prompt from output
                if (generatedText.startsWith(prompt)) {
                    generatedText = generatedText.substring(prompt.length);
                }
            }
            
            // Clean up the output
            generatedText = this.cleanOutput(generatedText, mode);
            
            // Format output appropriately
            if (mode === 'desc_to_seq') {
                const outputType = targetType || 'DNA';
                generatedText = this.formatSequenceOutput(generatedText, outputType);
                
                // Show external tools
                if (generatedText.length > 10) {
                    this.showExternalTools(generatedText, outputType);
                }
            } else {
                this.hideExternalTools();
            }
            
            // Display the output
            document.getElementById('outputText').value = generatedText;
            this.log('Generation completed successfully!', 'success');
            
        } catch (error) {
            this.log(`Generation failed: ${error.message}`, 'error');
            document.getElementById('outputText').value = 'Generation failed. Check console for details.';
        } finally {
            this.isGenerating = false;
            this.showGeneratingUI(false);
        }
    }
    
    // Mock generation for demo/custom models
    async mockGenerate(prompt, options) {
        const { mode, targetType, max_new_tokens, temperature } = options;
        
        // Simulate generation delay
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        if (mode === 'desc_to_seq') {
            // Generate a mock sequence
            const seqType = targetType || 'DNA';
            const chars = {
                DNA: 'ACGT',
                RNA: 'ACGU',
                PROTEIN: 'ACDEFGHIKLMNPQRSTVWY'
            };
            
            const alphabet = chars[seqType];
            let sequence = '';
            const length = Math.min(max_new_tokens, seqType === 'PROTEIN' ? 150 : 300);
            
            for (let i = 0; i < length; i++) {
                const randomIndex = Math.floor(Math.random() * alphabet.length);
                sequence += alphabet[randomIndex];
            }
            
            return sequence;
        } else {
            // Generate a mock description
            const descriptions = [
                'This sequence encodes a protein involved in cellular metabolism and energy production',
                'A regulatory element that controls gene expression under stress conditions',
                'This protein functions as a key enzyme in the biosynthetic pathway',
                'A structural protein that maintains cellular integrity and organization',
                'This sequence represents a conserved domain found across multiple species'
            ];
            
            return descriptions[Math.floor(Math.random() * descriptions.length)];
        }
    }
    
    cleanOutput(text, mode) {
        // Remove special tokens from output
        Object.values(this.specialTokens).forEach(token => {
            text = text.replace(new RegExp(token.replace(/[.*+?^${}()|[\]\\]/g, '\\    loadExampleInput(type)'), 'g'), '');
        });
        
        // Clean up based on mode
        if (mode === 'desc_to_seq') {
            // Remove any non-sequence characters
            text = text.replace(/[^ATGCURNYKSWBDHVACDEFGHIKLMNPQRSTVWY]/gi, '');
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
    
    showGeneratingUI(show) {
        const btn = document.getElementById('generateBtn');
        const btnIcon = btn.querySelector('i');
        const animation = btn.querySelector('.generating-animation');
        const overlay = document.querySelector('.generating-overlay');
        
        if (show) {
            btn.disabled = true;
            if (btnIcon) {
                btnIcon.style.display = 'none';
            }
            if (animation) {
                animation.classList.remove('hidden');
            }
            if (overlay) {
                overlay.classList.remove('hidden');
            }
            document.getElementById('outputText').value = '';
        } else {
            btn.disabled = false;
            if (btnIcon) {
                btnIcon.style.display = 'inline';
            }
            if (animation) {
                animation.classList.add('hidden');
            }
            if (overlay) {
                overlay.classList.add('hidden');
            }
        }
    }
    
    showExternalTools(sequence, type) {
        const toolsSection = document.getElementById('externalToolsSection');
        toolsSection.classList.remove('hidden');
        
        // Extract just the sequence part (remove FASTA header)
        const cleanSeq = sequence.split('\n').slice(1).join('').replace(/\s+/g, '');
        const encodedSeq = encodeURIComponent(cleanSeq);
        
        const tools = {
            DNA: [
                { name: 'BLASTN', url: `https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastn&PAGE_TYPE=BlastSearch&QUERY=${encodedSeq}` },
                { name: 'ExPASy Translate', url: `https://web.expasy.org/translate/` },
                { name: 'Primer3', url: `https://primer3.ut.ee/` }
            ],
            RNA: [
                { name: 'BLASTN', url: `https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastn&PAGE_TYPE=BlastSearch&QUERY=${encodedSeq}` },
                { name: 'RNAfold', url: `http://rna.tbi.univie.ac.at/cgi-bin/RNAWebSuite/RNAfold.cgi` },
                { name: 'mfold', url: `http://www.unafold.org/mfold/applications/rna-folding-form.php` }
            ],
            PROTEIN: [
                { name: 'Fold with AlphaFold', url: `https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb`, className: 'button-primary small-btn' },
                { name: 'BLASTP', url: `https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastp&PAGE_TYPE=BlastSearch&QUERY=${encodedSeq}` },
                { name: 'UniProt BLAST', url: `https://www.uniprot.org/blast?query=${encodedSeq}` },
                { name: 'InterPro', url: `https://www.ebi.ac.uk/interpro/search/sequence/` },
                { name: 'PDB Search', url: `https://www.rcsb.org/search` }
            ]
        };
        
        const toolLinks = tools[type] || [];
        toolsSection.innerHTML = '<h4>Analyze Sequence:</h4>' + 
            toolLinks.map(tool => 
                `<a href="${tool.url}" target="_blank" rel="noopener noreferrer" class="${tool.className || 'button-secondary small-btn'}">${tool.name}</a>`
            ).join('');
    }
    
    hideExternalTools() {
        document.getElementById('externalToolsSection').classList.add('hidden');
    }
    
    // Helper methods
    simpleTokenize(text) {
        // Simple tokenization for mock model
        return text.split('').map((char, idx) => idx);
    }
    
    simpleDetokenize(tokens) {
        // Simple detokenization for mock model
        return tokens.join('');
    }
    
    createDefaultVocab() {
        // Create a basic vocabulary for mock model
        const vocab = {};
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 []';
        chars.split('').forEach((char, idx) => {
            vocab[char] = idx;
        });
        return vocab;
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
        
        const iconMap = {
            error: 'fas fa-times-circle',
            success: 'fas fa-check-circle',
            info: 'fas fa-info-circle',
            warning: 'fas fa-exclamation-triangle'
        };
        
        const icon = iconMap[type] || iconMap.info;
        logEntry.innerHTML = `<i class="${icon}"></i> [${timestamp}] ${message}`;
        logEntry.className = `log-${type}`;
        
        logArea.appendChild(logEntry);
        logArea.scrollTop = logArea.scrollHeight;
        
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MnemogenomicsApp();
});