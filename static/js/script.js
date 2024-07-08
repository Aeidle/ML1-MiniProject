const socket = io('http://127.0.0.1:5000')

socket.on('update_data', function(data) {
    // Extract data for plotting
    let timestamps = data.face_data.map(d => d.timestamp);
    // let engagement = data.engagement_data.map(d => d.engagement);

    // let classes = data.engagement_bar.map(d => d.classes)
    let engagement_count = data.engagement_bar;

    let label = data.label;

    let earValues = data.ear_data.map(d => d.ear_value);
    let faceCounts = data.face_data.map(d => d.face_count);

    // Update the predicted label
    document.getElementById('predictedLabel').innerHTML = label;



    // Engagement Chart
    let engagementTrace = {
        x: ['writing_reading', 'distracted_mouth_open', 'using_smartphone', 'focused_mouth_closed', 
            'distracted_mouth_closed', 'fatigue', 'focused_mouth_open', 'raise_hand', 'listening', 'sleeping'],
        y: engagement_count,
        marker : {
            color : [
            'rgba(173, 216, 230, 1)',
            'rgba(135, 206, 235, 1)',
            'rgba(224, 255, 255, 1)',
            'rgba(70, 130, 180, 1)',
            'rgba(106, 90, 205, 1)',
            'rgba(0, 123, 167, 1)',
            'rgba(0, 128, 128, 1)',
            'rgba(176, 224, 230, 1)',
            'rgba(240, 255, 255, 1)',
            'rgba(176, 224, 230, 1)'
        ]
          },
        type: 'bar',
        // mode: 'lines+markers',
        name: 'Engagement'
    };
    
    let engagementLayout = {
        title: 'Engagement Over Time',
        xaxis: { title: 'Classes' },
        yaxis: { title: 'Count' }
    };

    Plotly.newPlot('engagementChart', [engagementTrace], engagementLayout);

    // Faces Detected Chart
    let faceTrace = {
        x: timestamps,
        y: faceCounts,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Faces Detected',
        fill: 'tonexty',
        line: {
            color: 'rgb(0, 0, 255)',
            width: 2,
            dash: 'solid',
            opacity: 1
        }
    };
    
    let faceLayout = {
        title: 'Faces Detected Over Time',
        xaxis: { title: 'Timestamp' },
        yaxis: { title: 'Faces Detected', fixedrange: true, range: [0, Math.max(...faceCounts) + 1] }
    };

    Plotly.newPlot('facesChart', [faceTrace], faceLayout);

    // EAR Chart
    let earTrace = {
        x: timestamps,
        y: earValues,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'EAR',
        fill: 'tozeroy',
        line: {
            color: 'rgb(255, 0, 0)',
            width: 2,
            dash: 'solid',
            opacity: 1
        }
    };
    
    let earLayout = {
        title: 'EAR Over Time',
        xaxis: { title: 'Timestamp' },
        yaxis: { title: 'EAR Value', fixedrange: true, range: [0, Math.max(...earValues) + 1] }
    };

    Plotly.newPlot('earChart', [earTrace], earLayout);
});


// socket.on('cluster_data', function(data) {
//     let student_group = data.group;
//     console.log("Received cluster group:", student_group);

//     // Update the predicted label
//     document.getElementById('cluster-group').textContent = student_group.toString(); // Ensure it's converted to string if necessary
// });

socket.on('connect_error', function(err) {
    console.error('Socket connection error:', err);
});

socket.on('disconnect', function() {
    console.log('Socket disconnected');
});


document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    form.addEventListener('submit', function (event) {
        event.preventDefault();

        // Collect form data
        const formData = new FormData(form);

        // Send form data to Flask endpoint
        fetch('/perform_clustering', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.group) {
                // Update DOM with the received group value
                const clusterGroup = document.getElementById('cluster-group');
                clusterGroup.textContent = data.group;

                console.log(data.X_original)

                updateClusterPlot(data.cluster_centers, data.X, data.data, data.group, data.X_original);
            } else {
                console.error('Error: Failed to retrieve clustering result');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});

function updateClusterPlot(clusterCenters, X, data, group, X_original) {
    // Extracting data for the new cluster
    const X_new = X;
    const cluster_centers_ = clusterCenters;
    const y_kmeans = data;
    const new_cluster_label = group;
    const X_all = X_original;

    // Create traces for each cluster
    let traces = [];
    let colors = ['red', 'blue', 'green'];

    for (let i = 0; i < 3; i++) {
        let trace = {
            x: X_all.filter((_, index) => y_kmeans[index] === i).map(a => a[0]),
            y: X_all.filter((_, index) => y_kmeans[index] === i).map(a => a[1]),
            z: X_all.filter((_, index) => y_kmeans[index] === i).map(a => a[2]),
            mode: 'markers',
            marker: {
                size: 5,
                color: colors[i],
                opacity: 0.8
            },
            type: 'scatter3d',
            name: `Cluster ${i + 1}`
        };
        traces.push(trace);
    }

    // Adding the new cluster trace
    let trace_new_cluster = {
        x: X_new.map(a => a[0]),
        y: X_new.map(a => a[1]),
        z: X_new.map(a => a[2]),
        mode: 'markers',
        marker: {
            size: 10,
            color: 'purple',  // Purple color for the new cluster
            opacity: 0.8
        },
        type: 'scatter3d',
        name: `New Cluster (${new_cluster_label})`
    };
    traces.push(trace_new_cluster);

    // Plotting centroids
    let trace_centroids = {
        x: [cluster_centers_[0][0], cluster_centers_[1][0], cluster_centers_[2][0]],
        y: [cluster_centers_[0][1], cluster_centers_[1][1], cluster_centers_[2][1]],
        z: [cluster_centers_[0][2], cluster_centers_[1][2], cluster_centers_[2][2]],
        mode: 'markers',
        marker: {
            size: 10,
            color: 'yellow',
            opacity: 0.8
        },
        type: 'scatter3d',
        name: 'Centroids'
    };

    let layout = {
        title: 'Cluster of Students Engagement (K-Means)',
        margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 0
        },
        scene: {
            xaxis: { title: 'Feature 1' },
            yaxis: { title: 'Feature 2' },
            zaxis: { title: 'Feature 3' }
        }
    };

    // Create a new Plotly chart with the traces and layout
    Plotly.newPlot('clusterPlot', traces.concat([trace_centroids]), layout);
}
