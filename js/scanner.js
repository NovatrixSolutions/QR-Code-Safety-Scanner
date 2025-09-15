// JavaScript for Scanner page

document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const mobileMenu = document.getElementById('mobileMenu');
    const navLinks = document.querySelector('.nav-links');
    
    if (mobileMenu) {
        mobileMenu.addEventListener('click', function() {
            navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
        });
    }
    
    // Elements
    const uploadArea = document.getElementById('uploadArea');
    const qrInput = document.getElementById('qrInput');
    const chooseFileBtn = document.getElementById('chooseFileBtn');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const clearImage = document.getElementById('clearImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsSection = document.getElementById('resultsSection');
    const resultCard = document.getElementById('resultCard');
    
    // Event Listeners
    chooseFileBtn.addEventListener('click', () => qrInput.click());
    
    qrInput.addEventListener('change', handleFileSelect);
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary-color)';
        uploadArea.style.backgroundColor = '#f8fafc';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--border-color)';
        uploadArea.style.backgroundColor = 'white';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border-color)';
        uploadArea.style.backgroundColor = 'white';
        
        if (e.dataTransfer.files.length) {
            qrInput.files = e.dataTransfer.files;
            handleFileSelect();
        }
    });
    
    clearImage.addEventListener('click', resetUpload);
    
    analyzeBtn.addEventListener('click', analyzeQRCode);
    
    // Functions
    function handleFileSelect() {
        if (!qrInput.files || !qrInput.files[0]) return;
        
        const file = qrInput.files[0];
        const validTypes = ['image/jpeg', 'image/png', 'image/gif'];
        
        if (!validTypes.includes(file.type)) {
            alert('Please select a valid image file (JPG, PNG, GIF)');
            resetUpload();
            return;
        }
        
        if (file.size > 10 * 1024 * 1024) {
            alert('File size exceeds 10MB limit');
            resetUpload();
            return;
        }
        
        const reader = new FileReader();
        
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            fileName.textContent = file.name;
            fileSize.textContent = formatFileSize(file.size);
            imagePreview.style.display = 'block';
            analyzeBtn.disabled = false;
        };
        
        reader.readAsDataURL(file);
    }
    
    function resetUpload() {
        qrInput.value = '';
        previewImg.src = '';
        imagePreview.style.display = 'none';
        analyzeBtn.disabled = true;
        resultsSection.style.display = 'none';
    }
    
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / 1048576).toFixed(1) + ' MB';
    }
    
    function analyzeQRCode() {
        // Show loading state
        analyzeBtn.querySelector('.loading-spinner').style.display = 'block';
        analyzeBtn.disabled = true;
        
        // Simulate analysis (in a real app, this would be an API call)
        setTimeout(() => {
            analyzeBtn.querySelector('.loading-spinner').style.display = 'none';
            analyzeBtn.disabled = false;
            
            // Generate fake analysis results
            const isSafe = Math.random() > 0.5;
            const threatLevel = isSafe ? 'safe' : (Math.random() > 0.5 ? 'suspicious' : 'malicious');
            
            displayResults(threatLevel);
        }, 2000);
    }
    
    function displayResults(threatLevel) {
        // Sample data for demonstration
        const sampleData = {
            safe: {
                status: 'Safe',
                icon: 'fa-check-circle',
                class: 'status-safe',
                confidence: '98%',
                url: 'https://example.com/safe-redirect',
                indicators: [
                    { type: 'Domain Age', status: 'safe', text: 'Registered for 3+ years' },
                    { type: 'SSL Certificate', status: 'safe', text: 'Valid HTTPS' },
                    { type: 'Content Analysis', status: 'safe', text: 'No malicious content' },
                    { type: 'Reputation', status: 'safe', text: 'Trusted website' }
                ],
                details: [
                    { label: 'Domain Created', value: 'March 12, 2019' },
                    { label: 'Server Location', value: 'United States' },
                    { label: 'Redirects', value: 'None detected' },
                    { label: 'Tracking', value: 'Minimal analytics' }
                ],
                recommendation: 'This QR code appears to be safe. You can proceed with caution.'
            },
            suspicious: {
                status: 'Suspicious',
                icon: 'fa-exclamation-triangle',
                class: 'status-warning',
                confidence: '65%',
                url: 'http://suspicious-site.xyz/offer?uid=283742',
                indicators: [
                    { type: 'Domain Age', status: 'warning', text: 'Registered 15 days ago' },
                    { type: 'SSL Certificate', status: 'danger', text: 'No HTTPS encryption' },
                    { type: 'Content Analysis', status: 'safe', text: 'No malicious content' },
                    { type: 'Reputation', status: 'warning', text: 'Unknown reputation' }
                ],
                details: [
                    { label: 'Domain Created', value: 'July 5, 2025' },
                    { label: 'Server Location', value: 'Offshore hosting' },
                    { label: 'Redirects', value: '2 redirects detected' },
                    { label: 'Tracking', value: 'Multiple tracking parameters' }
                ],
                recommendation: 'This QR code shows suspicious characteristics. Avoid entering any personal information.'
            },
            malicious: {
                status: 'Malicious',
                icon: 'fa-times-circle',
                class: 'status-danger',
                confidence: '92%',
                url: 'http://malicious-site.biz/login.php?redirect=paypal.com',
                indicators: [
                    { type: 'Domain Age', status: 'danger', text: 'Registered 2 days ago' },
                    { type: 'SSL Certificate', status: 'danger', text: 'No HTTPS encryption' },
                    { type: 'Content Analysis', status: 'danger', text: 'Phishing page detected' },
                    { type: 'Reputation', status: 'danger', text: 'Blacklisted domain' }
                ],
                details: [
                    { label: 'Domain Created', value: 'August 18, 2025' },
                    { label: 'Server Location', value: 'Unknown jurisdiction' },
                    { label: 'Redirects', value: '3 redirects to suspicious domain' },
                    { label: 'Tracking', value: 'Extensive user tracking' }
                ],
                recommendation: 'This QR code is highly likely to be malicious. Do not proceed and delete this QR code immediately.'
            }
        };
        
        const data = sampleData[threatLevel];
        
        // Build results HTML
        let html = `
            <div class="result-header">
                <div class="result-status">
                    <i class="fas ${data.icon} status-icon ${data.class}"></i>
                    <span>${data.status}</span>
                </div>
                <div class="result-confidence">Confidence: ${data.confidence}</div>
            </div>
            <div class="result-body">
                <a href="#" class="result-url">${data.url}</a>
                
                <div class="threat-indicators">
                    <h3>Security Indicators</h3>
                    <div class="indicator-grid">
        `;
        
        data.indicators.forEach(indicator => {
            html += `
                <div class="indicator-item">
                    <i class="fas ${indicator.status === 'safe' ? 'fa-check-circle indicator-safe' : 
                                  indicator.status === 'warning' ? 'fa-exclamation-triangle indicator-warning' : 
                                  'fa-times-circle indicator-danger'} indicator-icon"></i>
                    <div>
                        <strong>${indicator.type}:</strong> ${indicator.text}
                    </div>
                </div>
            `;
        });
        
        html += `
                    </div>
                </div>
                
                <div class="result-details">
                    <h3>Detailed Analysis</h3>
        `;
        
        data.details.forEach(detail => {
            html += `
                <div class="detail-item">
                    <span class="detail-label">${detail.label}</span>
                    <span>${detail.value}</span>
                </div>
            `;
        });
        
        html += `
                </div>
                
                <div class="recommendation">
                    <h3>Security Recommendation</h3>
                    <p>${data.recommendation}</p>
                </div>
            </div>
        `;
        
        resultCard.innerHTML = html;
        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
});