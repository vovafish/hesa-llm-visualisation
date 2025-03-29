/**
 * HESA Data Visualization - AI Dashboard JavaScript
 * 
 * This file contains the JavaScript code for the AI-powered dashboard.
 */

// Add the getCsrfToken function at the top of the file
function getCsrfToken() {
    // Django puts the CSRF token in a cookie named csrftoken
    const cookieValue = document.cookie
        .split('; ')
        .find(row => row.startsWith('csrftoken='))
        ?.split('=')[1];
        
    if (cookieValue) {
        return cookieValue;
    }
    
    // If not in cookies, get from the hidden csrf input field that Django provides
    const csrfElement = document.querySelector('input[name="csrfmiddlewaretoken"]');
    if (csrfElement) {
        return csrfElement.value;
    }
    
    console.error('CSRF token not found. This may cause API requests to fail.');
    return '';
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('🤖 AI Dashboard JavaScript loaded');
    
    // Core elements
    const aiQueryInput = document.getElementById('aiQueryInput');
    const aiMaxMatches = document.getElementById('aiMaxMatches');
    const aiSearchBtn = document.getElementById('aiSearchBtn');
    const aiQueryResults = document.getElementById('aiQueryResults');
    const aiSampleQueriesBtn = document.getElementById('aiSampleQueriesBtn');
    const aiSampleQueriesDropdown = document.getElementById('aiSampleQueriesDropdown');
    
    // Log element availability for debugging
    console.log('Elements found:', {
        aiQueryInput: !!aiQueryInput,
        aiMaxMatches: !!aiMaxMatches,
        aiSearchBtn: !!aiSearchBtn,
        aiQueryResults: !!aiQueryResults,
        aiSampleQueriesBtn: !!aiSampleQueriesBtn,
        aiSampleQueriesDropdown: !!aiSampleQueriesDropdown
    });
    
    // Hide any existing loading indicators on page load
    hideLoading();
    
    // Sample Queries Button Click Handler
    if (aiSampleQueriesBtn && aiSampleQueriesDropdown) {
        aiSampleQueriesBtn.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent event from bubbling up
            aiSampleQueriesDropdown.classList.toggle('hidden');
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            if (!aiSampleQueriesBtn.contains(e.target) && !aiSampleQueriesDropdown.contains(e.target)) {
                aiSampleQueriesDropdown.classList.add('hidden');
            }
        });
        
        // Handle sample query item clicks
        const sampleQueryItems = document.querySelectorAll('.ai-sample-query-item');
        sampleQueryItems.forEach(item => {
            item.addEventListener('click', function() {
                const query = this.getAttribute('data-query');
                if (aiQueryInput && query) {
                    aiQueryInput.value = query;
                    
                    // Optional: Highlight the input to indicate it was changed
                    aiQueryInput.classList.add('ring-2', 'ring-blue-500');
                    setTimeout(() => {
                        aiQueryInput.classList.remove('ring-2', 'ring-blue-500');
                    }, 1000);
                    
                    aiSampleQueriesDropdown.classList.add('hidden');
                }
            });
        });
    }
    
    // AI Search Button Click Handler
    if (aiSearchBtn && aiQueryInput) {
        aiSearchBtn.addEventListener('click', function() {
            const query = aiQueryInput.value.trim();
            
            console.log('AI Search button clicked with query:', query);
            
            if (!query) {
                alert('Please enter a query');
                return;
            }
            
            // Show loading state
            showLoading('Analyzing your query with Gemini AI...');
            
            // Prepare max matches parameter
            const maxMatches = aiMaxMatches ? aiMaxMatches.value : 3;
            
            // Call the Gemini API endpoint
            fetch('/process_gemini_query/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken(),
                },
                body: JSON.stringify({
                    query: query,
                    max_matches: maxMatches
                })
            })
            .then(response => {
                console.log('Received response with status:', response.status);
                
                if (!response.ok) {
                    return response.text().then(text => {
                        console.error('Error response body:', text);
                        throw new Error('Network response was not ok: ' + response.status);
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('Received data from API:', data);
                hideLoading();
                
                if (data.status === 'error') {
                    console.error('Error from backend:', data.error);
                    showError(data.error);
                    return;
                }
                
                // Display the query analysis
                displayQueryAnalysis(data);
            })
            .catch(error => {
                console.error('Fetch error:', error);
                hideLoading();
                showError('An error occurred while processing your query. Please try again.');
            });
        });
    }
    
    // Function to display query analysis
    function displayQueryAnalysis(data) {
        console.log('Displaying query analysis:', data);
        
        if (!aiQueryResults) {
            console.error('Query results container not found');
            return;
        }
        
        // Check if there was an API error
        if (data.api_error) {
            aiQueryResults.innerHTML = `
                <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mt-4" role="alert">
                    <p class="font-bold">Gemini API Error</p>
                    <p>${data.api_error}</p>
                    <p class="mt-2 text-sm">Using fallback analysis instead. For accurate entity extraction, please ensure the Gemini API key is configured correctly.</p>
                </div>
            `;
            
            // We'll still show the fallback results below the error message
        }
        
        // Format the year range if available
        let yearRangeDisplay = 'Not specified';
        if (data.start_year) {
            if (data.end_year && data.start_year !== data.end_year) {
                yearRangeDisplay = `${data.start_year} to ${data.end_year}`;
            } else {
                yearRangeDisplay = data.start_year;
            }
        }
        
        // Format data request categories
        let dataRequestDisplay = 'General information';
        if (data.data_request && data.data_request.length > 0 && data.data_request[0] !== 'general_data') {
            dataRequestDisplay = data.data_request.map(item => item.replace('_', ' ')).join(', ');
        }
        
        // Create HTML for the query analysis
        let resultsHTML = `
            <div class="bg-white shadow-md rounded-lg p-6 mt-8">
                <h3 class="text-lg font-semibold mb-4">
                    Query Analysis ${data.using_mock ? '(Mock AI)' : 'by Gemini AI'}
                    ${data.using_mock ? 
                        '<span class="text-xs font-normal bg-yellow-100 text-yellow-800 px-2 py-1 rounded ml-2">Using offline AI simulation</span>' : 
                        '<span class="text-xs font-normal bg-green-100 text-green-800 px-2 py-1 rounded ml-2">Using Google Gemini API</span>'}
                </h3>
                
                <div class="mb-4">
                    <p class="font-medium text-gray-700">Your query:</p>
                    <p class="text-blue-800 bg-blue-50 p-2 rounded">${data.query}</p>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="border rounded p-3">
                        <p class="font-medium text-gray-700">Institutions:</p>
                        <ul class="list-disc list-inside">
                            ${data.institutions && data.institutions.length ? 
                                data.institutions.map(inst => `<li>${inst}</li>`).join('') : 
                                '<li class="text-gray-500">None specified</li>'}
                        </ul>
                    </div>
                    
                    <div class="border rounded p-3">
                        <p class="font-medium text-gray-700">Years:</p>
                        <ul class="list-disc list-inside">
                            ${data.years && data.years.length ? 
                                data.years.map(year => `<li>${year}</li>`).join('') : 
                                '<li class="text-gray-500">All available years</li>'}
                        </ul>
                    </div>
                    
                    <div class="border rounded p-3">
                        <p class="font-medium text-gray-700">Year Range:</p>
                        <p class="${data.start_year ? 'font-semibold text-blue-700' : 'text-gray-500'}">
                            ${yearRangeDisplay}
                        </p>
                    </div>
                </div>
                
                <div class="mt-4 border rounded p-3">
                    <p class="font-medium text-gray-700">Data requested:</p>
                    <p class="font-semibold ${dataRequestDisplay === 'General information' ? 'text-gray-500' : 'text-blue-700'}">
                        ${dataRequestDisplay}
                    </p>
                </div>
                
                <div class="mt-4 text-sm text-gray-600 border-t pt-2">
                    <p>Entity extraction performed by ${data.using_mock ? 
                        'local AI simulation (Gemini API unavailable)' : 
                        'Google\'s Gemini AI'}</p>
                </div>
            </div>
        `;
        
        // If matching grouped datasets are available, display them
        if (data.grouped_datasets && data.grouped_datasets.length > 0) {
            resultsHTML += `
                <div class="bg-white shadow-md rounded-lg p-6 mt-8">
                    <h3 class="text-lg font-semibold mb-4">
                        Matching Datasets <span class="text-blue-600">(${data.grouped_datasets.length})</span>
                        <span class="text-xs font-normal bg-blue-100 text-blue-800 px-2 py-1 rounded ml-2">
                            Semantically matched by ${data.using_mock ? 'Regex' : 'Gemini AI'}
                        </span>
                    </h3>
                    <p class="text-sm text-gray-600 mb-4">The following datasets were found to be relevant to your query:</p>
                    <div class="space-y-4">
            `;
            
            // Add each grouped dataset
            data.grouped_datasets.forEach((dataset, index) => {
                // Handle score display
                const matchScore = dataset.score !== undefined ? parseFloat(dataset.score).toFixed(2) : 'N/A';
                const matchPercentage = dataset.match_percentage || Math.round(parseFloat(matchScore) * 100) || 'N/A';
                
                // Determine score color based on match quality
                let scoreColorClass = 'bg-gray-100 text-gray-800';
                if (matchScore !== 'N/A') {
                    const score = parseFloat(matchScore);
                    if (score >= 0.8) {
                        scoreColorClass = 'bg-green-100 text-green-800';
                    } else if (score >= 0.5) {
                        scoreColorClass = 'bg-blue-100 text-blue-800';
                    } else if (score >= 0.3) {
                        scoreColorClass = 'bg-yellow-100 text-yellow-800';
                    } else {
                        scoreColorClass = 'bg-orange-100 text-orange-800';
                    }
                }
                
                // Format academic years for display
                const academicYearsDisplay = dataset.academic_years ? 
                    dataset.academic_years.join(', ') : 
                    (dataset.academic_year || 'Unknown');
                
                // Format reference files for display
                const referencesDisplay = dataset.references ? 
                    dataset.references.map(ref => `<div class="text-xs bg-gray-50 p-1 my-1 rounded border">${ref}</div>`).join('') : 
                    (dataset.reference || 'Unknown');
                
                // Combine all descriptions
                const combinedDescription = dataset.descriptions ? 
                    dataset.descriptions.join('<br><br>') : 
                    (dataset.description || '');
                
                resultsHTML += `
                    <div class="border rounded-lg p-4 hover:bg-blue-50 transition-colors">
                        <div class="flex justify-between items-start">
                            <h4 class="font-semibold text-blue-800 text-lg">${index + 1}. ${dataset.title || 'Untitled Dataset'}</h4>
                            <span class="text-sm px-2 py-1 rounded-full ${scoreColorClass}">
                                Match: ${matchScore} (${matchPercentage}%)
                            </span>
                        </div>
                        
                        <div class="mt-2 text-gray-600 text-sm">
                            <div class="px-2 py-1 bg-gray-100 rounded mb-2">
                                <span class="font-medium">Academic Years:</span> ${academicYearsDisplay}
                            </div>
                            
                            <div class="px-2 py-1 bg-gray-100 rounded">
                                <span class="font-medium">References:</span>
                                <div class="mt-1">${referencesDisplay}</div>
                            </div>
                        </div>
                        
                        ${dataset.matched_terms && dataset.matched_terms.length > 0 ? `
                            <div class="mt-3">
                                <span class="font-medium text-sm text-gray-700">Matched Terms:</span>
                                <div class="flex flex-wrap gap-1 mt-1">
                                    ${dataset.matched_terms.map(term => 
                                        `<span class="text-xs px-2 py-1 bg-blue-50 text-blue-700 rounded-full">${term}</span>`
                                    ).join('')}
                                </div>
                            </div>
                        ` : ''}
                        
                        ${combinedDescription ? `
                            <div class="mt-3 text-sm text-gray-700 bg-gray-50 p-2 rounded">
                                <span class="font-medium">Why this matches:</span>
                                <div class="mt-1">${combinedDescription}</div>
                            </div>
                        ` : ''}
                        
                        <div class="mt-3">
                            <div class="border-t pt-3">
                                <span class="font-medium text-sm">Available Files:</span>
                                <div class="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2">
                                    ${dataset.matches ? dataset.matches.map(match => `
                                        <div class="border rounded p-2 bg-white">
                                            <div class="text-sm font-medium">${match.academic_year}</div>
                                            <div class="text-xs text-gray-600 mb-2">${match.reference}</div>
                                            <a href="/dataset/${encodeURIComponent(match.reference)}" 
                                               class="text-xs bg-blue-600 hover:bg-blue-700 text-white py-1 px-2 rounded inline-flex items-center">
                                                <svg class="h-3 w-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
                                                </svg>
                                                View
                                            </a>
                                        </div>
                                    `).join('') : ''}
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            resultsHTML += `
                    </div>
                </div>
            `;
        } else if (data.matching_datasets && data.matching_datasets.length === 0) {
            // No matching datasets found
            resultsHTML += `
                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4 mt-8">
                    <div class="flex">
                        <div class="flex-shrink-0">
                            <svg class="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                            </svg>
                        </div>
                        <div class="ml-3">
                            <p class="text-sm text-yellow-700">
                                No matching datasets found for your query. Try adjusting your search terms or time period.
                            </p>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Update the query results container
        aiQueryResults.innerHTML = resultsHTML;
    }
    
    // Function to show loading indicator
    function showLoading(message = 'Processing your query...') {
        console.log('Showing loading indicator');
        
        // Check if loading indicator already exists
        let loadingIndicator = document.getElementById('aiLoadingIndicator');
        
        if (!loadingIndicator) {
            // Create loading indicator if it doesn't exist
            loadingIndicator = document.createElement('div');
            loadingIndicator.id = 'aiLoadingIndicator';
            loadingIndicator.className = 'fixed top-0 left-0 w-full h-full flex items-center justify-center bg-gray-800 bg-opacity-50 z-50';
            loadingIndicator.innerHTML = `
                <div class="bg-white p-5 rounded-lg shadow-lg flex flex-col items-center">
                    <div class="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-blue-500 mb-3"></div>
                    <p class="text-gray-700">${message}</p>
                </div>
            `;
            
            document.body.appendChild(loadingIndicator);
            console.log('Loading indicator created and added to page');
        } else {
            // Show existing loading indicator
            loadingIndicator.classList.remove('hidden');
        }
    }
    
    // Function to hide loading indicator
    function hideLoading() {
        console.log('Hiding loading indicator');
        
        const loadingIndicator = document.getElementById('aiLoadingIndicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
            console.log('Loading indicator removed');
        }
    }
    
    // Function to show error message
    function showError(message) {
        console.log('Showing error message:', message);
        
        if (!aiQueryResults) {
            console.error('Query results container not found');
            return;
        }
        
        aiQueryResults.innerHTML = `
            <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mt-4" role="alert">
                <p class="font-bold">Error</p>
                <p>${message}</p>
            </div>
        `;
    }
}); 