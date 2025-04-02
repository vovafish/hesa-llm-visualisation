/**
 * HESA Data Visualization - AI Dashboard JavaScript
 * 
 * This file contains the JavaScript code for the AI-powered dashboard.
 */

// Function to get CSRF token from cookies
function getCsrfToken() {
    const cookieValue = document.cookie
        .split('; ')
        .find(row => row.startsWith('csrftoken='))
        ?.split('=')[1];
    
    if (cookieValue) {
        return cookieValue;
    }
    
    // If no CSRF token in cookies, try to get it from the page meta tag
    const csrfInput = document.querySelector('input[name="csrfmiddlewaretoken"]');
    if (csrfInput) {
        return csrfInput.value;
    }
    
    // Return empty string if no CSRF token found
    return '';
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('🤖 AI Dashboard JavaScript loaded');
    
    // Initialize loading state
    let isLoading = false;
    
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
    
    // Function to clean column headers and data
    function cleanMetadataArtifacts(columns, tableData) {
        console.log('Cleaning metadata artifacts from columns:', columns);
        
        // Handle case where no columns are provided
        if (!columns || columns.length === 0) {
            console.warn('No columns provided to cleanMetadataArtifacts');
            return { columns: [], data: tableData || [] };
        }
        
        // Remove any metadata string from column headers
        const cleanColumns = columns.map(col => {
            // Remove #METADATA and any JSON that follows it
            if (typeof col === 'string') {
                // Handle metadata in column headers
                const metadataRemoved = col.replace(/#METADATA:.*?}/g, '').trim();
                
                // Make column names look nicer (capitalize first letter of each word)
                return metadataRemoved.split(/[\s_]+/).map(word => 
                    word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
                ).join(' ');
            }
            return col;
        });
        
        console.log('Columns after metadata removal:', cleanColumns);
        
        // Remove duplicate "Academic Year" columns
        const uniqueColumns = [];
        const seenColumns = new Set();
        
        cleanColumns.forEach(col => {
            if (!seenColumns.has(col)) {
                uniqueColumns.push(col);
                seenColumns.add(col);
            } else {
                console.log(`Duplicate column removed: ${col}`);
            }
        });
        
        console.log('Final unique columns:', uniqueColumns);
        
        // Clean row data - remove any metadata strings and ensure correct length
        const cleanData = (tableData || []).map(row => {
            // Handle metadata in row data
            const cleanRow = row.map(cell => {
                if (typeof cell === 'string') {
                    return cell.replace(/#METADATA:.*?}/g, '').trim();
                }
                return cell;
            });
            
            // Adjust row length to match columns
            if (cleanRow.length > uniqueColumns.length) {
                console.log(`Trimming row from ${cleanRow.length} to ${uniqueColumns.length} columns`);
                return cleanRow.slice(0, uniqueColumns.length);
            } else if (cleanRow.length < uniqueColumns.length) {
                console.log(`Padding row from ${cleanRow.length} to ${uniqueColumns.length} columns`);
                const paddedRow = [...cleanRow];
                while (paddedRow.length < uniqueColumns.length) {
                    paddedRow.push('');
                }
                return paddedRow;
            }
            return cleanRow;
        });
        
        return { columns: uniqueColumns, data: cleanData };
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
                
                ${data.has_institution_typos || data.has_year_typos ? `
                <div class="mb-4 bg-blue-50 p-3 rounded border border-blue-200">
                    <p class="font-medium text-gray-700">Corrected query:</p>
                    <p class="text-blue-800 p-2">${generateCorrectedQuery(data)}</p>
                </div>
                ` : ''}
                
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
        
        // Add warning for missing years if applicable
        if (data.missing_years && data.missing_years.length > 0) {
            resultsHTML += `
                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4 mt-4" role="alert">
                    <div class="flex">
                        <div class="flex-shrink-0">
                            <svg class="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                            </svg>
                        </div>
                        <div class="ml-3">
                            <p class="text-sm text-yellow-700">
                                <span class="font-bold">Note:</span> Data for the following academic years was requested but is not available: 
                                ${data.missing_years.join(', ')}
                            </p>
                        </div>
                    </div>
                </div>
            `;
        }
        
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
                    <div class="space-y-4" id="datasets-list">
                        <!-- Datasets will be inserted here -->
                        <div class="text-center p-4">
                            <div class="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mx-auto"></div>
                            <p class="mt-2 text-gray-600">Loading datasets...</p>
                        </div>
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
        addDataAttributes(data);
        
        // Process datasets in a separate step
        if (data.grouped_datasets && data.grouped_datasets.length > 0) {
            setTimeout(() => {
                const datasetsList = document.getElementById('datasets-list');
                if (!datasetsList) {
                    console.error("Datasets list container not found");
                    return;
                }
                
                // Clear loading indicator
                datasetsList.innerHTML = '';
                
                // Add each dataset
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
                    
                    const datasetHTML = `
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
                                    <div class="mt-2" id="files-container-${index}">
                                        <div class="text-center p-2">
                                            <div class="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-blue-500 mx-auto"></div>
                                            <p class="mt-1 text-gray-600 text-sm">Loading file previews...</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="mt-4 pt-2 border-t">
                                <button class="select-dataset-btn bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 w-full" 
                                    data-dataset-title="${dataset.title || ''}" 
                                    data-dataset-references='${JSON.stringify(dataset.references || [])}'>
                                    Select this dataset
                                </button>
                            </div>
                        </div>
                    `;
                    
                    datasetsList.innerHTML += datasetHTML;
                    
                    // Load file previews in a separate step
                    setTimeout(() => {
                        renderFilePreview(dataset, index);
                    }, 10);
                });
            }, 0);
        }
    }
    
    // Function to render file previews for a dataset
    function renderFilePreview(dataset, datasetIndex) {
        console.log(`Rendering file previews for dataset ${datasetIndex + 1}: ${dataset.title || 'Untitled'}`);
        
        const filesContainer = document.getElementById(`files-container-${datasetIndex}`);
        if (!filesContainer) {
            console.error(`Files container not found for dataset ${datasetIndex}`);
            return;
        }
        
        if (!dataset.matches || dataset.matches.length === 0) {
            filesContainer.innerHTML = `
                <div class="text-sm text-gray-500 p-3 bg-gray-50 rounded">
                    No files available
                </div>
            `;
            return;
        }
        
        let filesHTML = '';
        
        dataset.matches.forEach(match => {
            console.log(`Processing match for dataset ${datasetIndex + 1}:`, match);
            
            filesHTML += `
                <div class="border rounded p-3 bg-white my-4">
                    <div class="text-sm font-medium mb-2">${match.academic_year || 'Unknown Year'} - ${match.reference || 'Unknown Reference'}</div>
            `;
            
            if (match.preview && match.preview.columns && match.preview.data) {
                console.log(`Processing preview data for ${match.reference}`, {
                    columns: match.preview.columns.length,
                    data: match.preview.data.length
                });
                
                try {
                    // Clean metadata artifacts from columns and data
                    const { columns: uniqueColumns, data: previewData } = cleanMetadataArtifacts(
                        match.preview.columns, 
                        match.preview.data
                    );
                    
                    filesHTML += `
                        <div class="overflow-x-auto max-h-[300px] table-container border rounded">
                            <table class="min-w-full border-collapse table-auto text-sm">
                                <thead>
                                    <tr>
                                        ${uniqueColumns.map(column => 
                                            `<th class="px-4 py-2 border-b border-gray-300 text-left text-sm font-medium sticky top-0 bg-white z-10">${column}</th>`
                                        ).join('')}
                                    </tr>
                                </thead>
                                <tbody>
                                    ${previewData.length > 0 ? 
                                        previewData.map(row => `
                                            <tr>
                                                ${row.map(cell => 
                                                    `<td class="px-4 py-2 border-b border-gray-300 text-sm">${cell}</td>`
                                                ).join('')}
                                            </tr>
                                        `).join('') : 
                                        `<tr><td colspan="${uniqueColumns.length}" class="px-4 py-2 text-center text-gray-500">No data available or no matching institutions found</td></tr>`
                                    }
                                </tbody>
                            </table>
                        </div>
                        ${match.preview.has_more ? 
                            `<div class="text-xs text-gray-500 mt-2">
                                Showing ${previewData.length} of ${match.preview.matched_rows || previewData.length} matching rows
                            </div>` : 
                            ''
                        }
                    `;
                } catch (error) {
                    console.error('Error processing preview data:', error);
                    filesHTML += `
                        <div class="text-sm text-red-500 p-3 bg-red-50 rounded">
                            Error processing preview data: ${error.message}
                        </div>
                    `;
                }
            } else if (match.preview && match.preview.error) {
                filesHTML += `
                    <div class="text-sm text-red-500 p-3 bg-red-50 rounded">
                        Error loading preview: ${match.preview.error}
                    </div>
                `;
            } else {
                filesHTML += `
                    <div class="text-sm text-gray-500 p-3 bg-gray-50 rounded">
                        Preview not available
                    </div>
                `;
            }
            
            filesHTML += `</div>`;
        });
        
        filesContainer.innerHTML = filesHTML;
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
    
    // Function to generate corrected query display text
    function generateCorrectedQuery(data) {
        let query = data.query;
        
        // Replace institution names with corrected versions
        if (data.has_institution_typos && data.original_institutions && data.institutions) {
            for (let i = 0; i < data.original_institutions.length; i++) {
                if (i < data.institutions.length && data.original_institutions[i] !== data.institutions[i]) {
                    // Create a case-insensitive regex to find all instances
                    const regex = new RegExp(data.original_institutions[i], 'gi');
                    query = query.replace(regex, `<span class="text-green-600 font-medium">${data.institutions[i]}</span>`);
                }
            }
        }
        
        // Replace years with corrected versions
        if (data.has_year_typos && data.original_years && data.years) {
            for (let i = 0; i < data.original_years.length; i++) {
                if (i < data.years.length && data.original_years[i] !== data.years[i]) {
                    // Create a case-insensitive regex to find all instances
                    const regex = new RegExp(data.original_years[i], 'gi');
                    query = query.replace(regex, `<span class="text-green-600 font-medium">${data.years[i]}</span>`);
                }
            }
        }
        
        return query;
    }
    
    // Function to handle Select Dataset button clicks
    document.addEventListener('click', function(e) {
        if (e.target && e.target.classList.contains('select-dataset-btn')) {
            const datasetTitle = e.target.getAttribute('data-dataset-title');
            const datasetReferences = JSON.parse(e.target.getAttribute('data-dataset-references') || '[]');
            
            if (!datasetTitle || !datasetReferences.length) {
                console.error('Missing dataset information for selection');
                return;
            }
            
            console.log('Selected dataset:', datasetTitle);
            console.log('References:', datasetReferences);
            
            // Show loading state
            showLoading('Loading complete dataset...');
            
            // Gather institutions from the query analysis
            const queryResults = document.getElementById('aiQueryResults');
            let institutions = [];
            let originalInstitutions = [];
            let query = '';
            let correctedQuery = '';
            
            // Try to extract this information from the data attributes on the page
            const institutionsElem = queryResults.querySelector('[data-institutions]');
            const originalInstitutionsElem = queryResults.querySelector('[data-original-institutions]');
            const queryElem = queryResults.querySelector('[data-query]');
            const correctedQueryElem = queryResults.querySelector('[data-corrected-query]');
            
            if (institutionsElem) {
                try {
                    institutions = JSON.parse(institutionsElem.getAttribute('data-institutions'));
                } catch (e) {
                    console.error('Error parsing institutions:', e);
                }
            }
            
            if (originalInstitutionsElem) {
                try {
                    originalInstitutions = JSON.parse(originalInstitutionsElem.getAttribute('data-original-institutions'));
                } catch (e) {
                    console.error('Error parsing original institutions:', e);
                }
            }
            
            if (queryElem) {
                query = queryElem.getAttribute('data-query') || '';
            }
            
            if (correctedQueryElem) {
                correctedQuery = correctedQueryElem.getAttribute('data-corrected-query') || '';
            }
            
            // Call the AI dataset details endpoint
            fetch('/ai_dataset_details/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken(),
                },
                body: JSON.stringify({
                    dataset_title: datasetTitle,
                    dataset_references: datasetReferences,
                    institutions: institutions,
                    original_institutions: originalInstitutions,
                    query: query,
                    corrected_query: correctedQuery
                })
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => {
                        throw new Error('Network response was not ok: ' + response.status + ' ' + text);
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('Received dataset details:', data);
                hideLoading();
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                // Instead of using URL parameters, use a form POST submission
                const form = document.createElement('form');
                form.method = 'POST';
                form.action = '/ai_dataset_details/';
                form.style.display = 'none';
                
                // Add CSRF token
                const csrfToken = getCsrfToken();
                const csrfInput = document.createElement('input');
                csrfInput.type = 'hidden';
                csrfInput.name = 'csrfmiddlewaretoken';
                csrfInput.value = csrfToken;
                form.appendChild(csrfInput);
                
                // Add the data as a hidden field
                const dataInput = document.createElement('input');
                dataInput.type = 'hidden';
                dataInput.name = 'dataset_data';
                dataInput.value = JSON.stringify(data);
                form.appendChild(dataInput);
                
                // Add form to document and submit
                document.body.appendChild(form);
                form.submit();
            })
            .catch(error => {
                console.error('Error fetching dataset details:', error);
                hideLoading();
                showError('An error occurred while fetching dataset details. Please try again.');
            });
        }
    });
    
    // Add data attributes to store query analysis data for later use
    function addDataAttributes(data) {
        const queryResults = document.getElementById('aiQueryResults');
        if (!queryResults) return;
        
        // Generate corrected query if needed
        let correctedQuery = '';
        if (data.has_institution_typos || data.has_year_typos) {
            correctedQuery = generateCorrectedQueryText(data);
        }
        
        // Add hidden spans with data attributes
        const dataElements = document.createElement('div');
        dataElements.style.display = 'none';
        dataElements.innerHTML = `
            <span data-institutions='${JSON.stringify(data.institutions || [])}' id="data-institutions"></span>
            <span data-original-institutions='${JSON.stringify(data.original_institutions || [])}' id="data-original-institutions"></span>
            <span data-query='${data.query || ""}' id="data-query"></span>
            <span data-corrected-query='${correctedQuery}' id="data-corrected-query"></span>
        `;
        
        // Remove any existing data elements
        const existingDataElements = queryResults.querySelector('#data-elements-container');
        if (existingDataElements) {
            existingDataElements.remove();
        }
        
        // Add new data elements
        dataElements.id = 'data-elements-container';
        queryResults.appendChild(dataElements);
    }
    
    // Function to generate plain text version of corrected query (without HTML)
    function generateCorrectedQueryText(data) {
        let query = data.query;
        
        // Replace institution names with corrected versions
        if (data.has_institution_typos && data.original_institutions && data.institutions) {
            for (let i = 0; i < data.original_institutions.length; i++) {
                if (i < data.institutions.length && data.original_institutions[i] !== data.institutions[i]) {
                    // Create a case-insensitive regex to find all instances
                    const regex = new RegExp(data.original_institutions[i], 'gi');
                    query = query.replace(regex, data.institutions[i]);
                }
            }
        }
        
        // Replace years with corrected versions
        if (data.has_year_typos && data.original_years && data.years) {
            for (let i = 0; i < data.original_years.length; i++) {
                if (i < data.years.length && data.original_years[i] !== data.years[i]) {
                    // Create a case-insensitive regex to find all instances
                    const regex = new RegExp(data.original_years[i], 'gi');
                    query = query.replace(regex, data.years[i]);
                }
            }
        }
        
        return query;
    }
}); 