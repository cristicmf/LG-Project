// 主脚本文件

// 页面加载完成后执行
window.onload = function() {
    // 加载系统状态
    loadSystemStatus();
    
    // 绑定邮件表单提交事件
    document.getElementById('email-form').addEventListener('submit', function(e) {
        e.preventDefault();
        submitEmail();
    });
    
    // 绑定评估按钮点击事件
    document.getElementById('evaluate-btn').addEventListener('click', function() {
        evaluateAgent();
    });
    
    // 绑定折叠按钮点击事件，实现图标切换
    const collapseButtons = document.querySelectorAll('[data-bs-toggle="collapse"]');
    collapseButtons.forEach(button => {
        const targetId = button.getAttribute('data-bs-target');
        const targetElement = document.querySelector(targetId);
        
        // 初始化图标状态
        updateCollapseIcon(button, targetElement);
        
        // 绑定点击事件
        button.addEventListener('click', function() {
            // 延迟更新图标，确保 Bootstrap 的 collapse 事件已经触发
            setTimeout(() => {
                updateCollapseIcon(button, targetElement);
            }, 300);
        });
    });
};

// 更新折叠图标
function updateCollapseIcon(button, targetElement) {
    const iconElement = button.querySelector('.collapse-icon');
    if (iconElement) {
        if (targetElement.classList.contains('show')) {
            iconElement.textContent = '▼'; // 展开状态
        } else {
            iconElement.textContent = '▶'; // 折叠状态
        }
    }
}

// 加载系统状态
function loadSystemStatus() {
    fetch('/api/system-status')
        .then(response => response.json())
        .then(data => {
            // 显示系统信息
            const systemInfo = document.getElementById('system-info');
            systemInfo.innerHTML = `
                <p>时间: ${data.system_status.timestamp}</p>
                <p>系统状态: ${data.system_status.status}</p>
                <p>CPU 使用率: ${data.system_status.cpu_usage}%</p>
                <p>内存使用率: ${data.system_status.memory_usage}%</p>
            `;
            
            // 显示性能指标
            const performanceMetrics = document.getElementById('performance-metrics');
            let metricsHtml = '<ul>';
            for (const [key, value] of Object.entries(data.performance_metrics)) {
                if (typeof value === 'number') {
                    metricsHtml += `<li>${key}: ${value.toFixed(2)}ms</li>`;
                } else {
                    metricsHtml += `<li>${key}: ${value}</li>`;
                }
            }
            metricsHtml += '</ul>';
            performanceMetrics.innerHTML = metricsHtml;
            
            // 显示错误计数
            const errorCounts = document.getElementById('error-counts');
            let errorsHtml = '<ul>';
            for (const [key, value] of Object.entries(data.error_counts)) {
                errorsHtml += `<li>${key}: ${value}</li>`;
            }
            errorsHtml += '</ul>';
            errorCounts.innerHTML = errorsHtml;
        })
        .catch(error => {
            console.error('加载系统状态失败:', error);
        });
}

// 提交邮件请求
function submitEmail() {
    const senderEmail = document.getElementById('sender-email').value;
    const emailContent = document.getElementById('email-content').value;
    const emailResult = document.getElementById('email-result');
    
    // 显示加载状态
    emailResult.innerHTML = '<div class="alert alert-info">处理中...</div>';
    
    // 发送请求
    fetch('/api/submit-email', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            sender_email: senderEmail,
            email_content: emailContent
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const result = data.result;
            // 显示成功结果
            let html = `
                <div class="alert alert-success">邮件处理成功</div>
            `;
            
            // 显示邮件分类结果
            if (result.classification) {
                html += `
                    <h5>邮件分类结果:</h5>
                    <ul>
                        <li>意图: ${result.classification.intent}</li>
                        <li>优先级: ${result.classification.priority}</li>
                        <li>主题: ${result.classification.subject || '未分类'}</li>
                    </ul>
                `;
            }
            
            // 显示模拟回复的邮件信息
            if (result.draft_response) {
                html += `
                    <h5>模拟回复内容:</h5>
                    <div class="card">
                        <div class="card-header">
                            回复邮件
                        </div>
                        <div class="card-body">
                            <pre>${result.draft_response}</pre>
                        </div>
                    </div>
                `;
            }
            
            // 显示处理状态
            if (result.messages) {
                html += `
                    <h5>处理状态:</h5>
                    <ul>
                `;
                result.messages.forEach(msg => {
                    html += `<li>${msg}</li>`;
                });
                html += `
                    </ul>
                `;
            }
            
            // 显示完整结果（可选）
            html += `
                <h5>完整处理结果:</h5>
                <pre>${JSON.stringify(result, null, 2)}</pre>
            `;
            
            emailResult.innerHTML = html;
        } else {
            // 显示错误信息
            emailResult.innerHTML = `<div class="alert alert-danger">错误: ${data.error}</div>`;
        }
    })
    .catch(error => {
        console.error('提交邮件失败:', error);
        emailResult.innerHTML = `<div class="alert alert-danger">提交失败: ${error.message}</div>`;
    });
}

// 评估邮件代理
function evaluateAgent() {
    const evaluationResult = document.getElementById('evaluation-result');
    
    // 显示加载状态
    evaluationResult.innerHTML = '<div class="alert alert-info">评估中...</div>';
    
    // 发送请求
    fetch('/api/evaluate-agent', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 显示评估结果
            evaluationResult.innerHTML = `
                <div class="alert alert-success">评估成功</div>
                <h5>评估指标:</h5>
                <pre>${JSON.stringify(data.results, null, 2)}</pre>
                <h5>详细结果:</h5>
                <pre>${JSON.stringify(data.results, null, 2)}</pre>
            `;
        } else {
            // 显示错误信息
            evaluationResult.innerHTML = `<div class="alert alert-danger">错误: ${data.error}</div>`;
        }
    })
    .catch(error => {
        console.error('评估邮件代理失败:', error);
        evaluationResult.innerHTML = `<div class="alert alert-danger">评估失败: ${error.message}</div>`;
    });
}
