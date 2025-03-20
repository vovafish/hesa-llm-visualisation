/**
 * HESA Data Visualization - Dashboard JavaScript
 * 
 * Simplified version with basic query handling.
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard JavaScript loaded');
    
    // Get chart buttons
    const lineChartBtn = document.getElementById('lineChartBtn');
    const barChartBtn = document.getElementById('barChartBtn');
    const pieChartBtn = document.getElementById('pieChartBtn');
    
    // Get result container
    const queryResultsContainer = document.getElementById('queryResults');
    
    // Get sample queries elements
    const sampleQueriesBtn = document.getElementById('sampleQueriesBtn');
    const sampleQueriesDropdown = document.getElementById('sampleQueriesDropdown');
    
    // Get query input fields
    const queryInput = document.getElementById('queryInput');
    const institutionInput = document.getElementById('institutionInput');
    const startYearInput = document.getElementById('startYearInput');
    const endYearInput = document.getElementById('endYearInput');
    const maxMatchesInput = document.getElementById('maxMatches');
    
    console.log('Elements found:', {
        lineChartBtn: !!lineChartBtn,
        barChartBtn: !!barChartBtn,
        pieChartBtn: !!pieChartBtn,
        queryResultsContainer: !!queryResultsContainer,
        sampleQueriesBtn: !!sampleQueriesBtn,
        sampleQueriesDropdown: !!sampleQueriesDropdown,
        queryInput: !!queryInput,
        institutionInput: !!institutionInput,
        startYearInput: !!startYearInput,
        endYearInput: !!endYearInput,
        maxMatchesInput: !!maxMatchesInput
    });
    
    // Sample Queries Button Click Handler
    if (sampleQueriesBtn && sampleQueriesDropdown) {
        sampleQueriesBtn.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent event from bubbling up
            sampleQueriesDropdown.classList.toggle('hidden');
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            if (!sampleQueriesBtn.contains(e.target) && !sampleQueriesDropdown.contains(e.target)) {
                sampleQueriesDropdown.classList.add('hidden');
            }
        });
        
        // Handle sample query item clicks
        const sampleQueryItems = document.querySelectorAll('.sample-query-item');
        sampleQueryItems.forEach(item => {
            item.addEventListener('click', function() {
                if (queryInput && institutionInput && startYearInput && endYearInput) {
                    // Get data from the sample query button
                    const query = this.getAttribute('data-query');
                    const institution = this.getAttribute('data-institution');
                    const startYear = this.getAttribute('data-start-year');
                    const endYear = this.getAttribute('data-end-year');
                    
                    // Set the input values
                    queryInput.value = query || '';
                    institutionInput.value = institution || '';
                    startYearInput.value = startYear || '';
                    endYearInput.value = endYear || '';
                    
                    sampleQueriesDropdown.classList.add('hidden');
                    
                    // Optional: Highlight the inputs to indicate they were changed
                    [queryInput, institutionInput, startYearInput, endYearInput].forEach(input => {
                        if (input.value) {
                            input.classList.add('ring-2', 'ring-blue-500');
                            setTimeout(() => {
                                input.classList.remove('ring-2', 'ring-blue-500');
                            }, 1000);
                        }
                    });
                }
            });
        });
    }
    
    // Line Chart Button Click Handler
    lineChartBtn.addEventListener('click', function() {
        console.log('Line Chart button clicked');
        
        // Check if query input field is filled
        if (!queryInput || !queryInput.value.trim()) {
            alert('Please enter a data type query.');
            return;
        }
        
        // Show loading state
        showLoading();
        
        // Process the query with all input values
        processQuery('line');
    });
    
    // Bar Chart Button Click Handler
    barChartBtn.addEventListener('click', function() {
        console.log('Bar Chart button clicked');
        
        // Check if query input field is filled
        if (!queryInput || !queryInput.value.trim()) {
            alert('Please enter a data type query.');
            return;
        }
        
        alert('Bar chart functionality not implemented yet');
    });
    
    // Pie Chart Button Click Handler
    pieChartBtn.addEventListener('click', function() {
        console.log('Pie Chart button clicked');
        
        // Check if query input field is filled
        if (!queryInput || !queryInput.value.trim()) {
            alert('Please enter a data type query.');
            return;
        }
        
        alert('Pie chart functionality not implemented yet');
    });
    
    // Function to process the query and fetch data
    function processQuery(chartType) {
        // Get values from all input fields
        const query = queryInput.value.trim();
        const institution = institutionInput.value.trim();
        const startYear = startYearInput.value.trim();
        const endYear = endYearInput.value.trim();
        const maxMatches = maxMatchesInput.value || 3;
        
        console.log('Processing query:', {
            query,
            institution,
            startYear,
            endYear,
            chartType,
            maxMatches
        });
        
        // Validate inputs
        if (!query) {
            alert('Please enter a data type query.');
            hideLoading();
            return;
        }
        
        // If start year is provided, end year is required
        if (startYear && !endYear) {
            alert('If you provide a start year, you must also provide an end year.');
            hideLoading();
            return;
        }
        
        // Get the CSRF token for POST requests
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
        console.log('CSRF token found:', !!csrfToken);
        
        // Build URL parameters
        const params = new URLSearchParams({
            'query': query,
            'chart_type': chartType,
            'max_matches': maxMatches
        });
        
        // Add optional parameters if they exist
        if (institution) params.append('institution', institution);
        if (startYear) params.append('start_year', startYear);
        if (endYear) params.append('end_year', endYear);
        
        // Send the query to the backend
        console.log('Sending fetch request to /api/process-hesa-query/');
        console.log('Request URL parameters:', params.toString());
        
        fetch(`/api/process-hesa-query/?${params.toString()}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            }
        })
        .then(response => {
            console.log('Received response with status:', response.status);
            console.log('Response headers:', Object.fromEntries([...response.headers]));
            
            if (!response.ok) {
                // Log the error response text if possible
                return response.text().then(text => {
                    console.error('Error response body:', text);
                    
                    // Try to parse the error as JSON
                    let errorMessage = 'Network response was not ok: ' + response.status;
                    try {
                        const errorData = JSON.parse(text);
                        errorMessage = errorData.error || errorMessage;
                        
                        // Log additional debug info if available
                        if (errorData.debug_info) {
                            console.error('Debug info:', errorData.debug_info);
                        }
                        
                        // Display suggestions if available
                        if (errorData.suggestions && errorData.suggestions.length > 0) {
                            showError(errorMessage, errorData.suggestions);
                            return;
                        }
                    } catch (e) {
                        // Not JSON, use as-is
                        console.error('Could not parse error response as JSON');
                    }
                    
                    throw new Error(errorMessage);
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Received data:', data);
            hideLoading();
            
            if (data.status === 'error') {
                console.error('Error from backend:', data.error);
                showError(data.error);
                return;
            }
            
            // Display the query and results
            displayQueryResults(data);
        })
        .catch(error => {
            console.error('Fetch error:', error);
            hideLoading();
            showError('An error occurred while processing your query. Please try again.');
            console.error('Error:', error);
        });
    }
    
    // Function to display query results
    function displayQueryResults(data) {
        console.log('Displaying query results');
        
        // Clear previous results
        queryResultsContainer.innerHTML = '';
        
        // Check if we have preview data from the new structure
        if (data.preview_data && data.preview_data.length > 0) {
            console.log('Preview data found, displaying options');
            
            // Get input values for displaying in the info box
            const query = queryInput.value.trim();
            const institution = institutionInput.value.trim();
            const startYear = startYearInput.value.trim();
            const endYear = endYearInput.value.trim();
            
            // Construct a readable query description
            let queryDescription = `"${query}"`;
            if (institution) queryDescription += ` for ${institution}`;
            if (startYear && endYear) {
                queryDescription += ` (${startYear}-${endYear})`;
            }
            
            // Generate UI regarding removed words and added synonyms
            let removedWordsInfo = '';
            if (data.query_info && data.query_info.removed_words && data.query_info.removed_words.length > 0) {
                removedWordsInfo = `
                    <p class="mt-1 text-sm text-gray-600">
                        <span class="font-medium">Words removed from query:</span> ${data.query_info.removed_words.join(', ')}
                    </p>
                `;
            }
            
            let addedSynonymsInfo = '';
            if (data.query_info && data.query_info.added_synonyms && data.query_info.added_synonyms.length > 0) {
                addedSynonymsInfo = `
                    <p class="mt-1 text-sm text-gray-600">
                        <span class="font-medium">Synonyms added:</span> ${data.query_info.added_synonyms.join(', ')}
                    </p>
                `;
            }
            
            // Display info about multiple matches
            let infoHTML = `
                <div class="bg-blue-100 border-l-4 border-blue-500 text-blue-700 p-4 mb-6" role="alert">
                    <p class="font-bold">Your query matched ${data.preview_data.length} dataset types</p>
                    <p>Previews from each dataset are shown below. Select one to view complete results.</p>
                    <p class="mt-2 text-sm">Search query: ${queryDescription}</p>
                    <p class="mt-1 text-sm">Years: ${data.query_info.years && data.query_info.years.length > 0 ? data.query_info.years.join(", ") : "All available years"}</p>
                    ${removedWordsInfo}
                    ${addedSynonymsInfo}
                </div>
            `;
            queryResultsContainer.innerHTML = infoHTML;
            
            // Display each dataset preview
            data.preview_data.forEach(preview => {
                // Format group title
                const groupTitle = preview.title || "Unnamed Dataset";
                
                // Format match info
                const matchInfo = preview.matched_keywords || [];
                let matchInfoHtml = '';
                
                if (matchInfo.length > 0) {
                    if (matchInfo.length === 1 && matchInfo[0] === 'year match only') {
                        // Special case for year match only
                        matchInfoHtml = `
                            <div class="mt-2 text-sm">
                                <p class="font-medium text-yellow-700">Note: Only matched by year, no keyword matches found</p>
                            </div>
                        `;
                    } else {
                        // List actual matched keywords
                        matchInfoHtml = `
                            <div class="mt-2 text-sm">
                                <p class="font-medium">Matched keywords:</p>
                                <ul class="mt-1 list-disc list-inside text-gray-600">
                                    ${matchInfo.map(keyword => `<li>${keyword}</li>`).join('')}
                                </ul>
                            </div>
                        `;
                    }
                }
                
                // Format years info
                const availableYears = preview.available_years || [];
                const yearsText = availableYears.length > 0 ? 
                    `Available years: ${availableYears.join(', ')}` : 
                    'Year information not available';
                
                // Check if there are missing years
                let missingYearsHtml = '';
                if (preview.missing_years && preview.missing_years.length > 0) {
                    missingYearsHtml = `
                        <div class="mt-2 text-sm">
                            <p class="font-medium text-amber-700">Missing years from your request:</p>
                            <p class="text-amber-600">${preview.missing_years.join(', ')}</p>
                        </div>
                    `;
                }
                
                // Calculate color for match score based on percentage
                const matchPercent = parseFloat(preview.match_percentage);
                let matchScoreColor = 'text-gray-700';
                let matchScoreClass = '';
                if (matchPercent >= 50) {
                    matchScoreColor = 'text-green-700';
                    matchScoreClass = 'font-semibold';
                } else if (matchPercent >= 25) {
                    matchScoreColor = 'text-blue-700';
                } else if (matchPercent > 0) {
                    matchScoreColor = 'text-orange-700';
                } else {
                    matchScoreColor = 'text-gray-500';
                }
                
                // Create the group preview container
                // Change border color based on match quality
                let borderColorClass = 'border-gray-300';
                let bgColorClass = 'bg-gray-50';
                
                if (matchPercent >= 50) {
                    borderColorClass = 'border-green-500';
                    bgColorClass = 'bg-green-50';
                } else if (matchPercent >= 25) {
                    borderColorClass = 'border-blue-500';
                    bgColorClass = 'bg-blue-50';
                } else if (matchPercent > 0) {
                    borderColorClass = 'border-orange-500';
                    bgColorClass = 'bg-orange-50';
                }
                
                let groupPreviewHTML = `
                    <div class="rounded-lg shadow p-4 mb-6 group-preview border-l-4 ${borderColorClass} ${bgColorClass}">
                        <h3 class="text-xl font-semibold mb-2">${groupTitle}</h3>
                        <div class="grid grid-cols-2 gap-4 mb-3">
                            <p class="text-sm text-gray-700"><span class="font-medium">Files:</span> ${preview.file_count}</p>
                            <p class="text-sm ${matchScoreColor} ${matchScoreClass}"><span class="font-medium">Match score:</span> ${Math.round(preview.score * 10) / 10} (${preview.match_percentage}%)</p>
                            <p class="text-sm text-gray-700"><span class="font-medium">${yearsText}</span></p>
                        </div>
                        ${matchInfoHtml}
                        ${missingYearsHtml}
                        
                        <div class="mt-4 mb-2 font-medium text-green-700">File previews:</div>
                `;
                
                // Add each file preview
                if (preview.file_previews && preview.file_previews.length > 0) {
                    preview.file_previews.forEach((filePreview, index) => {
                        // Get filename and year
                        const fileName = filePreview.file_name;
                        const fileYear = filePreview.year || 'Unknown';
                        
                        groupPreviewHTML += `
                            <div class="mt-2 mb-4 pb-4 border-b border-green-300">
                                <h4 class="text-lg font-medium text-green-800">${fileName}</h4>
                                <p class="text-sm text-gray-600 mb-2">Year: ${fileYear}</p>
                                
                                <div class="overflow-x-auto mb-3">
                                    <table class="min-w-full bg-white border border-gray-300">
                                        <thead>
                                            <tr>
                        `;
                        
                        // Add table headers
                        const columns = filePreview.columns || [];
                        if (columns && columns.length > 0) {
                            columns.forEach(column => {
                                groupPreviewHTML += `<th class="px-4 py-2 border-b border-gray-300 text-left text-sm">${column}</th>`;
                            });
                        }
                        
                        groupPreviewHTML += `
                                            </tr>
                                        </thead>
                                        <tbody>
                        `;
                        
                        // Add table rows
                        const data = filePreview.data || [];
                        if (data && data.length > 0) {
                            // Show up to 3 rows in preview
                            const displayRows = data.slice(0, 3);
                            displayRows.forEach(row => {
                                groupPreviewHTML += `<tr>`;
                                row.forEach(cell => {
                                    groupPreviewHTML += `<td class="px-4 py-2 border-b border-gray-300 text-sm">${cell}</td>`;
                                });
                                groupPreviewHTML += `</tr>`;
                            });
                        } else {
                            groupPreviewHTML += `
                                <tr>
                                    <td colspan="${columns.length}" class="px-4 py-2 text-center text-gray-500">No matching rows found</td>
                                </tr>
                            `;
                        }
                        
                        groupPreviewHTML += `
                                        </tbody>
                                    </table>
                                </div>
                                
                                <!-- Preview information if there are more rows -->
                                ${filePreview.matched_rows > 3 ? 
                                `<div class="text-sm text-gray-600 mb-3">
                                    <p class="italic">Preview showing ${Math.min(data.length, 3)} rows out of ${filePreview.matched_rows}.</p>
                                </div>` : ''}
                            </div>
                        `;
                    });
                } else {
                    groupPreviewHTML += `
                        <div class="mt-2 mb-4">
                            <p class="text-gray-500">No file previews available.</p>
                        </div>
                    `;
                }
                
                // Add select button for the group
                groupPreviewHTML += `
                        <div class="flex justify-end mt-4">
                            <button class="select-file-btn bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded" data-group-id="${preview.group_id}">
                                Select this dataset
                            </button>
                        </div>
                    </div>
                `;
                
                queryResultsContainer.innerHTML += groupPreviewHTML;
            });
            
            // Add event listeners to select file buttons
            const selectFileBtns = document.querySelectorAll('.select-file-btn');
            selectFileBtns.forEach(btn => {
                btn.addEventListener('click', function() {
                    const groupId = this.getAttribute('data-group-id');
                    selectFileSource(groupId);
                });
            });
        } else {
            // No preview data
            queryResultsContainer.innerHTML = `
                <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4" role="alert">
                    <p class="font-bold">No data found</p>
                    <p>No data matched your query. Please try with different search terms.</p>
                </div>
            `;
        }
    }
    
    // Function to select a file source
    function selectFileSource(groupId) {
        console.log('Selecting file source with group ID:', groupId);
        
        // Get values from all input fields
        const query = queryInput.value.trim();
        const institution = institutionInput.value.trim();
        const startYear = startYearInput.value.trim();
        const endYear = endYearInput.value.trim();
        
        // Show loading state
        showLoading();
        
        // Get CSRF token
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
        
        // Send the selection to the backend
        fetch('/select_file_source/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': csrfToken,
            },
            body: new URLSearchParams({
                'group_id': groupId,
                'query': query,
                'institution': institution,
                'start_year': startYear,
                'end_year': endYear
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Received data from file selection:', data);
            hideLoading();
            
            if (data.status === 'error') {
                console.error('Error from backend:', data.error);
                showError(data.error);
                return;
            }
            
            // Display the full data
            displayFullData(data);
        })
        .catch(error => {
            console.error('Fetch error:', error);
            hideLoading();
            showError('An error occurred while selecting the data source. Please try again.');
        });
    }
    
    // Function to display the full data with visualization
    function displayFullData(data) {
        console.log('Displaying full data:', data);
        
        // Clear previous results
        queryResultsContainer.innerHTML = '';
        
        // Get group title and other info
        const groupTitle = data.group_title || 'Dataset';
        const institution = data.institution || 'All institutions';
        const filesData = data.files_data || [];
        
        if (filesData.length === 0) {
            queryResultsContainer.innerHTML = `
                <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4" role="alert">
                    <p class="font-bold">No data found</p>
                    <p>No data could be extracted from the selected dataset.</p>
                </div>
            `;
            return;
        }
        
        // Create header with dataset info
        let headerHTML = `
            <div class="bg-green-100 border-l-4 border-green-500 text-green-700 p-4 mb-6" role="alert">
                <p class="font-bold">Dataset: ${groupTitle}</p>
                <p>Showing data for ${institution}</p>
                <p class="mt-2 text-sm">Files: ${filesData.length}</p>
            </div>
        `;
        
        // Add file data sections
        let filesHTML = '';
        
        filesData.forEach(fileData => {
            const fileName = fileData.file_name;
            const fileYear = fileData.year || 'Unknown';
            const columns = fileData.columns || [];
            const data = fileData.data || [];
            
            filesHTML += `
                <div class="bg-white shadow-md rounded-lg p-4 mb-6">
                    <h3 class="text-lg font-semibold mb-2">${fileName}</h3>
                    <p class="text-sm text-gray-600 mb-4">Year: ${fileYear}</p>
                    
                    <div class="overflow-x-auto">
                        <table class="min-w-full bg-white border border-gray-300">
                            <thead>
                                <tr>
            `;
            
            // Add table headers
            columns.forEach(column => {
                filesHTML += `<th class="px-4 py-2 border-b border-gray-300 bg-gray-100 text-left">${column}</th>`;
            });
            
            filesHTML += `
                                </tr>
                            </thead>
                            <tbody>
            `;
            
            // Add table rows
            if (data.length > 0) {
                data.forEach(row => {
                    filesHTML += `<tr>`;
                    row.forEach(cell => {
                        filesHTML += `<td class="px-4 py-2 border-b border-gray-300">${cell}</td>`;
                    });
                    filesHTML += `</tr>`;
                });
            } else {
                filesHTML += `
                    <tr>
                        <td colspan="${columns.length}" class="px-4 py-2 text-center text-gray-500">No data available</td>
                    </tr>
                `;
            }
            
            filesHTML += `
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="mt-4 flex justify-end">
                        <button class="download-csv-btn bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded" 
                                data-filename="${fileName}" data-year="${fileYear}">
                            Download CSV
                        </button>
                    </div>
                </div>
            `;
        });
        
        // Combine all HTML
        queryResultsContainer.innerHTML = headerHTML + filesHTML;
        
        // Add event listeners for download buttons
        const downloadBtns = document.querySelectorAll('.download-csv-btn');
        downloadBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const fileName = this.getAttribute('data-filename');
                const fileYear = this.getAttribute('data-year');
                // Implement download functionality
                alert(`Download functionality for ${fileName} (${fileYear}) will be implemented soon.`);
            });
        });
    }
    
    // Function to show loading indicator
    function showLoading() {
        console.log('Showing loading indicator');
        
        // Check if loading indicator already exists
        let loadingIndicator = document.getElementById('loadingIndicator');
        
        if (!loadingIndicator) {
            // Create loading indicator if it doesn't exist
            loadingIndicator = document.createElement('div');
            loadingIndicator.id = 'loadingIndicator';
            loadingIndicator.className = 'fixed top-0 left-0 w-full h-full flex items-center justify-center bg-gray-800 bg-opacity-50 z-50';
            loadingIndicator.innerHTML = `
                <div class="bg-white p-5 rounded-lg shadow-lg flex flex-col items-center">
                    <div class="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-blue-500 mb-3"></div>
                    <p class="text-gray-700">Processing your query...</p>
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
        
        const loadingIndicator = document.getElementById('loadingIndicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
            console.log('Loading indicator removed');
        }
    }
    
    // Function to show error message
    function showError(message, suggestions = []) {
        console.log('Showing error message:', message);
        
        queryResultsContainer.innerHTML = `
            <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4" role="alert">
                <p class="font-bold">Error</p>
                <p>${message}</p>
            </div>
        `;
        
        if (suggestions.length > 0) {
            const suggestionsList = document.createElement('ul');
            suggestionsList.className = 'mt-2 list-disc list-inside text-sm text-gray-600';
            suggestions.forEach(suggestion => {
                const suggestionItem = document.createElement('li');
                suggestionItem.textContent = suggestion;
                suggestionsList.appendChild(suggestionItem);
            });
            queryResultsContainer.appendChild(suggestionsList);
        }
    }
});

function handleHesaResponse(response) {
    // DIRECT ALERT - This will definitely show up
    if (response && response.file_info) {
        let fileNames = [];
        if (Array.isArray(response.file_info)) {
            // Multiple files case
            fileNames = response.file_info.map(fi => fi.file_name || "Unknown file");
        } else {
            // Single file case (legacy)
            let fileName = "Unknown file";
            if (response.file_info.file_name) {
                fileName = response.file_info.file_name;
            } else if (response.file_info.cleaned_file_path) {
                fileName = response.file_info.cleaned_file_path.split('\\').pop();
            } else if (response.file_info.raw_file) {
                fileName = response.file_info.raw_file.split('\\').pop();
            }
            fileNames.push(fileName);
        }
        
        // Show an alert that can't be missed
        window.alert("FILES USED: " + fileNames.join(", "));
    }
    
    // Rest of the function continues...
    // Basic console log to ensure this function is being called
    console.log("⭐⭐⭐ HANDLE RESPONSE FUNCTION STARTED ⭐⭐⭐");
    
    if (response.status === 'success') {
        // Clear previous results
        document.getElementById('results-container').innerHTML = '';
        
        // Get raw file_info for debugging
        console.log("RAW FILE INFO:", JSON.stringify(response.file_info));
        
        // Extract filename(s) with fallbacks
        let fileInfo = "";
        if (Array.isArray(response.file_info)) {
            // Multiple files case
            fileInfo = response.file_info.map(fi => {
                return `${fi.year || 'Unknown year'}: ${fi.file_name || "Unknown file"}`;
            }).join("<br>");
        } else {
            // Single file case (legacy)
            let fileName = "Unknown file";
            if (response.file_info.file_name) {
                fileName = response.file_info.file_name;
            } else if (response.file_info.cleaned_file_path) {
                const parts = response.file_info.cleaned_file_path.split('\\');
                fileName = parts[parts.length - 1];
            } else if (response.file_info.raw_file) {
                const parts = response.file_info.raw_file.split('\\');
                fileName = parts[parts.length - 1];
            }
            fileInfo = fileName;
        }
        
        // Very visible console logs
        console.log("✅✅✅ FILES USED: " + fileInfo + " ✅✅✅");
        
        // Create a container for results
        const resultsContainer = document.getElementById('results-container');
        
        // 1. Create query section
        const querySection = document.createElement('div');
        querySection.className = 'mb-4';
        querySection.innerHTML = `
            <h3 class="text-lg font-bold mb-2">Query:</h3>
            <p>${response.query_info ? response.query_info.original_query || 'N/A' : 'N/A'}</p>
        `;
        resultsContainer.appendChild(querySection);
        
        // 2. Create a VERY visible filename section
        const fileNameSection = document.createElement('div');
        fileNameSection.style.backgroundColor = 'red';
        fileNameSection.style.color = 'white';
        fileNameSection.style.padding = '10px';
        fileNameSection.style.margin = '10px 0';
        fileNameSection.style.fontWeight = 'bold';
        fileNameSection.style.fontSize = '16px';
        fileNameSection.innerHTML = `FILES USED:<br>${fileInfo}`;
        resultsContainer.appendChild(fileNameSection);
        
        // 3. Results heading
        const resultsHeading = document.createElement('h3');
        resultsHeading.className = 'text-lg font-bold mb-2';
        resultsHeading.textContent = 'Results:';
        resultsContainer.appendChild(resultsHeading);
        
        // 4. Create table
        const table = document.createElement('table');
        table.className = 'min-w-full divide-y divide-gray-200';
        
        // Create header
        const thead = document.createElement('thead');
        thead.className = 'bg-gray-50';
        const headerRow = document.createElement('tr');
        
        if (response.columns && response.columns.length > 0) {
            response.columns.forEach(col => {
                const th = document.createElement('th');
                th.scope = 'col';
                th.className = 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
                th.textContent = col;
                headerRow.appendChild(th);
            });
        }
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Create body
        const tbody = document.createElement('tbody');
        tbody.className = 'bg-white divide-y divide-gray-200';
        
        if (response.data && response.data.length > 0) {
            response.data.forEach(row => {
                const dataRow = document.createElement('tr');
                response.columns.forEach(col => {
                    const td = document.createElement('td');
                    td.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
                    td.textContent = row[col] || '';
                    dataRow.appendChild(td);
                });
                tbody.appendChild(dataRow);
            });
        }
        
        table.appendChild(tbody);
        resultsContainer.appendChild(table);
        
        // 5. Create chart if chart data is available
        if (response.chart_data) {
            const chartSection = document.createElement('div');
            chartSection.className = 'mt-8';
            chartSection.innerHTML = `
                <h3 class="text-lg font-bold mb-2">Chart:</h3>
                <div class="h-96">
                    <canvas id="dataChart"></canvas>
                </div>
            `;
            resultsContainer.appendChild(chartSection);
            
            // Render chart
            const ctx = document.getElementById('dataChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: response.chart_data.labels,
                    datasets: response.chart_data.datasets || [{
                        label: response.chart_data.label || 'Value',
                        data: response.chart_data.values || [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }
        
        // Show the results container
        resultsContainer.classList.remove('hidden');
        
        // Final confirmation log
        console.log("⭐⭐⭐ HANDLE RESPONSE FUNCTION COMPLETED SUCCESSFULLY ⭐⭐⭐");
        
    } else {
        // Show error message
        showAlert('error', `Error: ${response.error}`);
    }
} 