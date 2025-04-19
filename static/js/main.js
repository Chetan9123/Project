// File Upload Handling
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.querySelector('.upload-area');
    const fileInput = document.querySelector('#file-input');
    
    if (uploadArea && fileInput) {
        // Drag and drop functionality
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            uploadArea.classList.add('dragover');
        }

        function unhighlight(e) {
            uploadArea.classList.remove('dragover');
        }

        uploadArea.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            fileInput.files = files;
            handleFiles(files);
        }

        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', function() {
            handleFiles(this.files);
        });
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (validateFile(file)) {
                showPreview(file);
                uploadFile(file);
            }
        }
    }

    function validateFile(file) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        const maxSize = 16 * 1024 * 1024; // 16MB

        if (!allowedTypes.includes(file.type)) {
            showAlert('Please upload a valid image file (JPEG, PNG)', 'danger');
            return false;
        }

        if (file.size > maxSize) {
            showAlert('File size should be less than 16MB', 'danger');
            return false;
        }

        return true;
    }

    function showPreview(file) {
        const reader = new FileReader();
        const preview = document.querySelector('#image-preview');
        
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.classList.remove('d-none');
        }
        
        reader.readAsDataURL(file);
    }

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        showSpinner();

        fetch('/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            if (data.success) {
                showResult(data);
            } else {
                showAlert(data.message, 'danger');
            }
        })
        .catch(error => {
            hideSpinner();
            showAlert('An error occurred during upload', 'danger');
            console.error('Error:', error);
        });
    }
});

// Alert Handling
function showAlert(message, type = 'info') {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type} alert-dismissible fade show`;
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const mainContainer = document.querySelector('main .container');
    mainContainer.insertBefore(alertContainer, mainContainer.firstChild);
    
    setTimeout(() => {
        alertContainer.remove();
    }, 5000);
}

// Loading Spinner
function showSpinner() {
    const spinner = document.createElement('div');
    spinner.className = 'spinner-overlay';
    spinner.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
    `;
    document.body.appendChild(spinner);
}

function hideSpinner() {
    const spinner = document.querySelector('.spinner-overlay');
    if (spinner) {
        spinner.remove();
    }
}

// Result Display
function showResult(data) {
    const resultContainer = document.querySelector('#result-container');
    if (!resultContainer) return;

    resultContainer.innerHTML = `
        <div class="card result-card fade-in">
            <div class="card-body">
                <span class="badge confidence-badge ${data.is_pothole ? 'bg-success' : 'bg-danger'}">
                    Confidence: ${(data.confidence * 100).toFixed(2)}%
                </span>
                <h5 class="card-title">${data.is_pothole ? 'Pothole Detected' : 'No Pothole Detected'}</h5>
                <p class="card-text">Location: ${data.location || 'Not specified'}</p>
                <div class="text-center">
                    <img src="${data.image_path}" class="img-fluid rounded" alt="Detection Result">
                </div>
            </div>
        </div>
    `;
}

// Map Integration
function initMap(lat, lng) {
    if (typeof google === 'undefined') return;
    
    const mapContainer = document.querySelector('.map-container');
    if (!mapContainer) return;

    const map = new google.maps.Map(mapContainer, {
        center: { lat: parseFloat(lat), lng: parseFloat(lng) },
        zoom: 15
    });

    new google.maps.Marker({
        position: { lat: parseFloat(lat), lng: parseFloat(lng) },
        map: map,
        title: 'Pothole Location'
    });
}

// Profile Image Upload
function handleProfileImageUpload(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.querySelector('.profile-avatar').src = e.target.result;
        };
        reader.readAsDataURL(input.files[0]);
    }
} 