/**
 * HESA Data Visualization - Dashboard JavaScript
 * 
 * Simplified version with basic query handling.
 */

// Add the getCsrfToken function at the top of the file, before document.addEventListener
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
    const allInstitutionsCheckbox = document.getElementById('allInstitutionsCheckbox');
    
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
        maxMatchesInput: !!maxMatchesInput,
        allInstitutionsCheckbox: !!allInstitutionsCheckbox
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
    
    // All institutions checkbox handler
    if (allInstitutionsCheckbox && institutionInput) {
        allInstitutionsCheckbox.addEventListener('change', function() {
            institutionInput.disabled = this.checked;
            if (this.checked) {
                institutionInput.classList.add('bg-gray-100');
            } else {
                institutionInput.classList.remove('bg-gray-100');
            }
        });
    }
    
    // Function to process the query and fetch data
    function processQuery(chartType) {
        // Get values from all input fields
        const query = queryInput.value.trim();
        let institution = institutionInput.value.trim();
        const startYear = startYearInput.value.trim();
        const endYear = endYearInput.value.trim();
        const maxMatches = maxMatchesInput.value || 3;
        const searchAllInstitutions = allInstitutionsCheckbox.checked;
        
        // Default to University of Leicester if no institution provided and checkbox is not checked
        if (!searchAllInstitutions) {
            if (!institution) {
                institution = "The University of Leicester";
            } else if (institution !== "The University of Leicester") {
                // If another institution is entered, include both it and Leicester
                institution = `The University of Leicester,${institution}`;
            }
        } else {
            // If checkbox is checked, don't filter by institution
            institution = "";
        }
        
        console.log('Processing query:', {
            query,
            institution,
            startYear,
            endYear,
            chartType,
            maxMatches,
            searchAllInstitutions
        });
        
        // Validate inputs
        if (!query) {
            alert('Please enter a data type query.');
            hideLoading();
            return;
        }
        
        // Note: We're removing the validation that requires end year when start year is provided
        // This allows searching for a single academic year when only start year is provided
        
        // Get CSRF token
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
        console.log('CSRF token found:', !!csrfToken);
        
        // Prepare the request parameters
        const params = new URLSearchParams();
        params.append('query', query);
        if (institution) params.append('institution', institution);
        if (startYear) params.append('start_year', startYear);
        if (endYear) params.append('end_year', endYear);
        params.append('chart_type', chartType);
        params.append('max_matches', maxMatches);
        
        // Build the API URL 
        const apiUrl = `/api/process-hesa-query/?${params.toString()}`;
        console.log('Fetching data from:', apiUrl);
        
        // Make the request to the backend
        fetch(apiUrl, {
            method: 'GET',
            headers: {
                'X-CSRFToken': csrfToken,
                'Content-Type': 'application/json'
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
        // Check if we have valid response data
        if (!data || !data.status || data.status !== 'success') {
            showError("Invalid response from server", ["Try a different query", "Contact system administrator"]);
            return;
        }
        
        // Store max preview rows value
        const maxPreviewRows = data.max_preview_rows || 3; // Default to 3 if not provided

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
            } else if (startYear) {
                queryDescription += ` (${startYear}/${startYear.toString().slice(-2)*1+1})`;  // e.g., 2015/16
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
                    // Add information about total files and preview limits
                    const totalFiles = preview.file_previews.length;
                    const initialFilesToShow = 3;
                    const hasMoreFiles = totalFiles > initialFilesToShow;
                    
                    if (hasMoreFiles) {
                        groupPreviewHTML += `
                            <div class="mb-2 text-sm text-gray-700">
                                <p>Showing ${initialFilesToShow} of ${totalFiles} files</p>
                            </div>
                        `;
                    }
                    
                    // Display initial previews (up to 3)
                    preview.file_previews.slice(0, initialFilesToShow).forEach((filePreview, index) => {
                        const columns = filePreview.columns;
                        const data = filePreview.data;
                        
                        // Show up to maxPreviewRows rows in preview
                        const displayRows = data.slice(0, maxPreviewRows);
                        
                        groupPreviewHTML += `
                            <div class="mt-2 mb-4 pb-4 border-b border-green-300">
                                <h4 class="text-lg font-medium text-green-800">${filePreview.file_name}</h4>
                                <p class="text-sm text-gray-600 mb-2">Year: ${filePreview.year || 'Unknown'}</p>
                                
                                <div class="overflow-x-auto mb-3">
                            <table class="min-w-full bg-white border border-gray-300">
                                <thead>
                                    <tr>
                                `;
                                
                // Add table headers
                        if (columns && columns.length > 0) {
                            columns.forEach(column => {
                                groupPreviewHTML += `<th class="px-4 py-2 border-b border-gray-300 text-left text-sm">${column}</th>`;
                            });
                        }
                        
                        // Add Academic Year header as the last column
                        groupPreviewHTML += `<th class="px-4 py-2 border-b border-gray-300 text-left text-sm">Academic Year</th>`;
                        
                        groupPreviewHTML += `
                                    </tr>
                                </thead>
                                <tbody>
                            `;
                
                // Add table rows
                        if (data && data.length > 0) {
                            // Show up to maxPreviewRows rows in preview
                            const displayRows = data.slice(0, maxPreviewRows);
                            
                            displayRows.forEach(row => {
                                groupPreviewHTML += `<tr>`;
                                row.forEach(cell => {
                                    groupPreviewHTML += `<td class="px-4 py-2 border-b border-gray-300 text-sm">${cell}</td>`;
                                });
                                // Add Academic Year cell to each row
                                groupPreviewHTML += `<td class="px-4 py-2 border-b border-gray-300 text-sm">${formatAcademicYear(filePreview.year) || ''}</td>`;
                                groupPreviewHTML += `</tr>`;
                            });
                        } else {
                            groupPreviewHTML += `
                                <tr>
                                    <td colspan="${columns.length + 1}" class="px-4 py-2 text-center text-gray-500">No matching rows found</td>
                                </tr>
                            `;
                        }
                        
                        groupPreviewHTML += `
                                </tbody>
                            </table>
                        </div>
                        
                        <!-- Preview information if there are more rows -->
                        ${filePreview.matched_rows > maxPreviewRows ? 
                        `<div class="text-sm text-gray-600 mb-3">
                            <p class="italic">Preview showing ${Math.min(data.length, maxPreviewRows)} rows out of ${filePreview.matched_rows}.</p>
                        </div>` : ''}
                            </div>
                        `;
                    });
                    
                    // Add hidden section for additional files if there are more than 3
                    if (hasMoreFiles) {
                        groupPreviewHTML += `<div class="hidden-files hidden">`;
                        
                        // Add remaining files (beyond the first 3)
                        preview.file_previews.slice(initialFilesToShow).forEach((filePreview, index) => {
                            const columns = filePreview.columns;
                            const data = filePreview.data;
                            
                            // Show up to maxPreviewRows rows in preview
                            const displayRows = data.slice(0, maxPreviewRows);
                            
                            groupPreviewHTML += `
                                <div class="mt-2 mb-4 pb-4 border-b border-green-300">
                                    <h4 class="text-lg font-medium text-green-800">${filePreview.file_name}</h4>
                                    <p class="text-sm text-gray-600 mb-2">Year: ${filePreview.year || 'Unknown'}</p>
                                    
                                    <div class="overflow-x-auto mb-3">
                                        <table class="min-w-full bg-white border border-gray-300">
                                            <thead>
                                                <tr>
                        `;
                        
                        // Add table headers
                        if (columns && columns.length > 0) {
                            columns.forEach(column => {
                                groupPreviewHTML += `<th class="px-4 py-2 border-b border-gray-300 text-left text-sm">${column}</th>`;
                            });
                        }
                        
                        // Add Academic Year header as the last column for hidden files too
                        groupPreviewHTML += `<th class="px-4 py-2 border-b border-gray-300 text-left text-sm">Academic Year</th>`;
                        
                        groupPreviewHTML += `
                                            </tr>
                                        </thead>
                                        <tbody>
                        `;
                        
                        // Add table rows
                        if (data && data.length > 0) {
                            // Show up to maxPreviewRows rows in preview
                            const displayRows = data.slice(0, maxPreviewRows);
                            
                            displayRows.forEach(row => {
                                groupPreviewHTML += `<tr>`;
                                row.forEach(cell => {
                                    groupPreviewHTML += `<td class="px-4 py-2 border-b border-gray-300 text-sm">${cell}</td>`;
                                });
                                // Add Academic Year cell to each row
                                groupPreviewHTML += `<td class="px-4 py-2 border-b border-gray-300 text-sm">${formatAcademicYear(filePreview.year) || ''}</td>`;
                                groupPreviewHTML += `</tr>`;
                            });
                        } else {
                            groupPreviewHTML += `
                                <tr>
                                    <td colspan="${columns.length + 1}" class="px-4 py-2 text-center text-gray-500">No matching rows found</td>
                                </tr>
                            `;
                        }
                        
                        groupPreviewHTML += `
                                        </tbody>
                                    </table>
                                </div>
                                
                                <!-- Preview information if there are more rows -->
                                ${filePreview.matched_rows > maxPreviewRows ? 
                                `<div class="text-sm text-gray-600 mb-3">
                                    <p class="italic">Preview showing ${Math.min(data.length, maxPreviewRows)} rows out of ${filePreview.matched_rows}.</p>
                                </div>` : ''}
                            </div>
                        `;
                        });
                        
                        groupPreviewHTML += `</div>`;
                        
                        // Add toggle button for showing/hiding additional files
                        groupPreviewHTML += `
                            <div class="mb-4">
                                <button class="toggle-files-btn text-blue-600 hover:text-blue-800 underline flex items-center">
                                    <span class="toggle-text">Show ${totalFiles - initialFilesToShow} more files</span>
                                    <svg xmlns="http://www.w3.org/2000/svg" class="toggle-icon h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                                    </svg>
                                </button>
                            </div>
                        `;
                    }
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
            
            // Set up click handlers for the select file buttons
            const selectFileBtns = document.querySelectorAll('.select-file-btn');
            selectFileBtns.forEach(btn => {
                btn.addEventListener('click', function() {
                    const groupId = this.getAttribute('data-group-id');
                    
                    // Get query parameters
                    const query = document.getElementById('queryInput').value;
                    const institution = document.getElementById('institutionInput').value;
                    const startYear = document.getElementById('startYearInput').value;
                    const endYear = document.getElementById('endYearInput').value;
                    
                    // Build redirect URL with query parameters
                    const url = new URL(`${window.location.origin}/dataset_details/${groupId}/`);
                    url.searchParams.append('query', query);
                    url.searchParams.append('institution', institution);
                    url.searchParams.append('start_year', startYear);
                    url.searchParams.append('end_year', endYear);
                    
                    // Redirect to the dataset details page
                    window.location.href = url.toString();
                });
            });

            // Set up toggle handlers for showing/hiding additional files
            queryResultsContainer.querySelectorAll('.toggle-files-btn').forEach(button => {
                button.addEventListener('click', function() {
                    // Find the closest hidden-files div
                    const hiddenFilesDiv = this.closest('.group-preview').querySelector('.hidden-files');
                    if (hiddenFilesDiv) {
                        // Toggle the hidden class
                        hiddenFilesDiv.classList.toggle('hidden');
                        
                        // Update the button text and icon
                        const toggleText = this.querySelector('.toggle-text');
                        const toggleIcon = this.querySelector('.toggle-icon');
                        
                        // Count the number of file preview containers inside the hidden section
                        // A better approach than using childElementCount which might count non-file elements
                        const totalHiddenFiles = hiddenFilesDiv.querySelectorAll('.mt-2.mb-4.pb-4').length;
                        
                        if (hiddenFilesDiv.classList.contains('hidden')) {
                            // Files are hidden, update text to show "Show more"
                            toggleText.textContent = `Show ${totalHiddenFiles} more files`;
                            // Rotate icon down
                            toggleIcon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />`;
                        } else {
                            // Files are shown, update text to show "Hide"
                            toggleText.textContent = 'Hide additional files';
                            // Rotate icon up
                            toggleIcon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7" />`;
                        }
                    }
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
    function selectFileSource(fileId) {
        // Show loading state
        showLoading();
        
        // Get the current query parameters
        const query = document.getElementById('queryInput').value;
        const institution = document.getElementById('institutionInput').value;
        const startYear = document.getElementById('startYearInput').value;
        const endYear = document.getElementById('endYearInput').value;
        
        console.log(`Selecting file source: ${fileId} with query: ${query}, institution: ${institution}, years: ${startYear}-${endYear}`);
        
        // Prepare the data
        const data = {
            query: query,
            institution: institution,
            startYear: startYear,
            endYear: endYear,
            fileId: fileId
        };
        
        // Get CSRF token - ensure this function is defined
        const csrfToken = getCsrfToken();
        if (!csrfToken) {
            showError('CSRF token not found. Please refresh the page and try again.');
            hideLoading();
            return;
        }
        
        // Send the request to select the file
        fetch('/select_file_source/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            console.log(`Response status: ${response.status}`);
            console.log(`Response headers:`, Object.fromEntries([...response.headers]));
            
            // Try to read the response as both JSON and text
            return response.text().then(text => {
                // Log the full response text for debugging
                console.log(`Response text (first 100 chars): ${text.substring(0, 100)}`);
                
                if (!response.ok) {
                    try {
                        // Try to parse as JSON first
                        const errorData = JSON.parse(text);
                        throw new Error(errorData.error || 'Error selecting file source');
                    } catch (e) {
                        // If parsing fails, return the raw text
                        throw new Error(`Server error: ${text}`);
                    }
                }
                
                try {
                    return JSON.parse(text);
                } catch (e) {
                    console.error("Error parsing JSON response:", e);
                    throw new Error(`Invalid response format: ${text.substring(0, 100)}...`);
                }
            });
        })
        .then(data => {
            // Hide the query results panel
            document.getElementById('queryResultsPanel').style.display = 'none';
            
            if (data.success) {
                console.log(`Received data with ${data.data.length} rows out of total ${data.total_row_count}`);
                
                // Display the full data in the visualization panel
                displayFullData(data);
                
                // Show the visualization panel
                document.getElementById('visualizationPanel').style.display = 'block';
                
                // Scroll to the visualization panel
                document.getElementById('visualizationPanel').scrollIntoView({
                    behavior: 'smooth'
                });
            } else {
                showError(`Error: ${data.error || 'Unknown error'}`);
                // Show query results panel again
                document.getElementById('queryResultsPanel').style.display = 'block';
            }
            
            hideLoading();
        })
        .catch(error => {
            console.error('Error selecting file source:', error);
            showError(`Error selecting file source: ${error.message}`, [
                'Check that the server is running',
                'Verify that the selected dataset exists',
                'Try refreshing the page and trying again',
                'Try selecting a different dataset from the results'
            ]);
            // Show query results panel again
            document.getElementById('queryResultsPanel').style.display = 'block';
            hideLoading();
        });
    }
    
    // Function to display the full data with visualization
    function displayFullData(data) {
        const visualizationPanel = document.getElementById('visualizationPanel');
        const dataContainer = document.getElementById('dataContainer');
        
        console.log('Displaying full data:', data);
        
        // Clear previous content
        dataContainer.innerHTML = '';
        
        // Check if we have valid data
        if (!data || !data.success) {
            dataContainer.innerHTML = `
                <div class="alert alert-danger">
                    <p class="font-bold">Error Loading Data</p>
                    <p>${data && data.error ? data.error : 'No data received from server'}</p>
                </div>
            `;
            return;
        }
        
        // Check if we have data
        if (!data.columns || !data.data || data.data.length === 0) {
            dataContainer.innerHTML = `
                <div class="alert alert-warning">
                    <p class="font-bold">No Data Available</p>
                    <p>No data available for the selected criteria. Try selecting a different dataset or modifying your search criteria.</p>
                </div>
            `;
            return;
        }
        
        // Create title with dataset title and row count information
        const titleDiv = document.createElement('div');
        titleDiv.className = 'mb-4';
        
        // Include dataset title if available
        const datasetTitle = data.dataset_title || 'Selected Dataset';
        
        // Include file count information if available
        const fileInfoText = data.file_count ? ` from ${data.file_count} file(s)` : '';
        
        titleDiv.innerHTML = `
            <h3 class="text-xl font-bold text-blue-800">${datasetTitle}</h3>
            <div class="mt-2">
                <span class="badge bg-primary px-2 py-1 rounded bg-blue-600 text-white">${data.total_row_count} rows${fileInfoText}</span>
                <p class="text-sm text-gray-600 mt-1">
                    Showing all matched data based on your query criteria.
                </p>
            </div>
        `;
        dataContainer.appendChild(titleDiv);
        
        // Create a responsive table container with maximum height and scrolling
        const tableContainer = document.createElement('div');
        tableContainer.className = 'overflow-auto max-h-[600px] border border-gray-300 rounded';
        
        // Create the table
        const table = document.createElement('table');
        table.className = 'min-w-full divide-y divide-gray-200';
        table.id = 'dataTable';
        
        // Create the table header with sticky position
        const thead = document.createElement('thead');
        thead.className = 'bg-gray-50 sticky top-0';
        const headerRow = document.createElement('tr');
        
        // Add headers
        data.columns.forEach(column => {
            const th = document.createElement('th');
            th.className = 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
            th.textContent = column;
            headerRow.appendChild(th);
        });
        
        // Add Academic Year header
        const academicYearHeader = document.createElement('th');
        academicYearHeader.className = 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
        academicYearHeader.textContent = 'Academic Year';
        headerRow.appendChild(academicYearHeader);
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Create the table body
        const tbody = document.createElement('tbody');
        tbody.className = 'bg-white divide-y divide-gray-200';
        
        // Add table rows
        if (data.data && data.data.length > 0) {
            data.data.forEach((row, rowIndex) => {
                const tr = document.createElement('tr');
                tr.className = rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50';
                
                // Add cells for each column
                row.forEach(cell => {
                    const td = document.createElement('td');
                    td.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-900';
                    td.textContent = cell;
                    tr.appendChild(td);
                });
                
                // Add academic year cell
                const academicYearCell = document.createElement('td');
                academicYearCell.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-900';
                academicYearCell.textContent = formatAcademicYear(data.academic_year) || '';
                tr.appendChild(academicYearCell);
                
                tbody.appendChild(tr);
            });
        } else {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-center';
            td.colSpan = data.columns.length + 1; // +1 for the academic year column
            td.textContent = 'No data available';
            tr.appendChild(td);
            tbody.appendChild(tr);
        }
        
        table.appendChild(tbody);
        tableContainer.appendChild(table);
        dataContainer.appendChild(tableContainer);
        
        // Show navigation/action buttons
        const actionDiv = document.createElement('div');
        actionDiv.className = 'mt-4 flex justify-between';
        
        const backButton = document.createElement('button');
        backButton.className = 'bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded';
        backButton.textContent = 'Back to Search Results';
        backButton.addEventListener('click', function() {
            // Hide visualization panel and show query results
            visualizationPanel.classList.add('hidden');
            document.getElementById('queryResultsPanel').classList.remove('hidden');
        });
        
        actionDiv.appendChild(backButton);
        dataContainer.appendChild(actionDiv);
        
        // Show the visualization panel
        visualizationPanel.classList.remove('hidden');
        document.getElementById('queryResultsPanel').classList.add('hidden');
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

// Format academic year to ensure YYYY/YY format
function formatAcademicYear(year) {
    if (!year) return '';
    
    // If already in YYYY/YY format, return as is
    if (/^\d{4}\/\d{2}$/.test(year)) {
        return year;
    }
    
    // Handle YYYY&YY format
    if (/^\d{4}&\d{2}$/.test(year)) {
        return year.replace('&', '/');
    }
    
    // Handle YYYY-YY format
    if (/^\d{4}-\d{2}$/.test(year)) {
        return year.replace('-', '/');
    }
    
    // Handle YYYY format - convert to YYYY/YY
    if (/^\d{4}$/.test(year)) {
        const startYear = parseInt(year);
        const endYearSuffix = String((startYear + 1) % 100).padStart(2, '0');
        return `${startYear}/${endYearSuffix}`;
    }
    
    // Return original if no pattern matches
    return year;
} 