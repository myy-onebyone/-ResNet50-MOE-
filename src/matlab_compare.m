%% 眼底疾病多标签分类 — ResNet50 vs MoE 对比实验可视化 (MATLAB)
%  读取Python导出的实验数据，生成6幅高质量对比图
clear; close all;

% ==================== 路径配置 ====================
% 获取脚本所在目录
scriptDir = fileparts(mfilename('fullpath'));
dataPath = fullfile(scriptDir, 'compare_results', 'comparison_data.mat');
outDir = fullfile(scriptDir, 'compare_results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% ==================== 加载数据 ====================
fprintf('加载数据: %s\n', dataPath);
data = load(dataPath);

% 类别名称
classNames = {'正常','糖尿病','青光眼','白内障','AMD','高血压','近视','其他疾病/异常'};

% 颜色方案
c1 = [0.13 0.59 0.22];  % 绿色 - ResNet50
c2 = [1.00 0.38 0.00];  % 橙色 - MoE

% ==================================================================
% 图1: 训练曲线对比
% ==================================================================
fprintf('[1/6] 训练曲线...\n');
figure('Position', [100, 100, 1400, 500], 'Color', 'w');

subplot(1,2,1);
epochs_rn = 1:length(data.rn_train_loss);
epochs_me = 1:length(data.me_train_loss);

h1 = plot(epochs_rn, data.rn_train_loss, '-o', 'Color', [c1 0.5], ...
          'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', c1);
hold on;
h2 = plot(epochs_rn, data.rn_val_loss, '-s', 'Color', c1, ...
          'LineWidth', 2.5, 'MarkerSize', 6, 'MarkerFaceColor', c1);
h3 = plot(epochs_me, data.me_train_loss, '-o', 'Color', [c2 0.5], ...
          'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', c2);
h4 = plot(epochs_me, data.me_val_loss, '-s', 'Color', c2, ...
          'LineWidth', 2.5, 'MarkerSize', 6, 'MarkerFaceColor', c2);
xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Loss', 'FontSize', 12, 'FontWeight', 'bold');
title('Loss 曲线对比', 'FontSize', 14, 'FontWeight', 'bold');
legend([h1, h2, h3, h4], ...
    {'ResNet50 训练', 'ResNet50 验证', 'MoE 训练', 'MoE 验证'}, ...
    'Location', 'northeast', 'FontSize', 9);
grid on; set(gca, 'FontSize', 10);

subplot(1,2,2);
plot(epochs_rn, data.rn_val_f1, '-o', 'Color', c1, ...
    'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', c1);
hold on;
plot(epochs_me, data.me_val_f1, '-s', 'Color', c2, ...
    'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', c2);

% 标注最优值
[best_rn, idx_rn] = max(data.rn_val_f1);
[best_me, idx_me] = max(data.me_val_f1);
plot(idx_rn, best_rn, 'v', 'Color', c1, 'MarkerSize', 14, ...
    'LineWidth', 2, 'MarkerFaceColor', c1);
plot(idx_me, best_me, '^', 'Color', c2, 'MarkerSize', 14, ...
    'LineWidth', 2, 'MarkerFaceColor', c2);
text(idx_rn+0.3, best_rn+0.008, sprintf('%.4f', best_rn), ...
    'Color', c1, 'FontWeight', 'bold', 'FontSize', 10);
text(idx_me+0.3, best_me-0.015, sprintf('%.4f', best_me), ...
    'Color', c2, 'FontWeight', 'bold', 'FontSize', 10);

xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Macro F1', 'FontSize', 12, 'FontWeight', 'bold');
title('验证集 Macro F1 曲线', 'FontSize', 14, 'FontWeight', 'bold');
legend({'ResNet50 (后融合)', 'MoE (门控融合)'}, ...
    'Location', 'southeast', 'FontSize', 10);
grid on; set(gca, 'FontSize', 10);

sgtitle('ResNet50 vs MoE — 训练曲线对比', ...
    'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(outDir, '01_training_curves.png'));
% 也保存为高清PNG
exportgraphics(gcf, fullfile(outDir, '01_training_curves.png'), ...
    'Resolution', 200);

% ==================================================================
% 图2: 总体指标对比
% ==================================================================
fprintf('[2/6] 总体指标...\n');
figure('Position', [100, 100, 1100, 650], 'Color', 'w');

metricLabels = {'Macro F1','Micro F1','Macro Precision','Macro Recall','Macro AUC'};
rnVals = [data.rn_macro_f1, data.rn_micro_f1, data.rn_macro_precision, ...
          data.rn_macro_recall, data.rn_macro_auc];
meVals = [data.me_macro_f1, data.me_micro_f1, data.me_macro_precision, ...
          data.me_macro_recall, data.me_macro_auc];

x = 1:length(metricLabels);
w = 0.32;

b1 = bar(x - w/2, rnVals, w, 'FaceColor', c1, 'EdgeColor', 'w', ...
    'LineWidth', 1.2, 'FaceAlpha', 0.9);
hold on;
b2 = bar(x + w/2, meVals, w, 'FaceColor', c2, 'EdgeColor', 'w', ...
    'LineWidth', 1.2, 'FaceAlpha', 0.9);

% 数值标注 + 差异
for i = 1:length(x)
    text(x(i)-w/2, rnVals(i)+0.012, sprintf('%.4f', rnVals(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    text(x(i)+w/2, meVals(i)+0.012, sprintf('%.4f', meVals(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    diff = meVals(i) - rnVals(i);
    if diff >= 0
        diffColor = [0.85 0.15 0.15];
        diffStr = sprintf('+%.4f', diff);
    else
        diffColor = [0.15 0.35 0.85];
        diffStr = sprintf('-%.4f', abs(diff));
    end
    yTop = max(rnVals(i), meVals(i)) + 0.07;
    text(x(i), yTop, ['\Delta=' diffStr], ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', ...
        'Color', diffColor, 'BackgroundColor', [1 1 0.9 0.7]);
end

set(gca, 'XTick', x, 'XTickLabel', metricLabels, 'FontSize', 11);
ylabel('Score', 'FontSize', 12, 'FontWeight', 'bold');
ylim([0, max([rnVals, meVals]) * 1.25]);
title('ResNet50 vs MoE — 总体指标对比', 'FontSize', 14, 'FontWeight', 'bold');
legend([b1, b2], {'ResNet50 (后融合)', 'MoE (门控融合)'}, ...
    'Location', 'northwest', 'FontSize', 11);
grid on; set(gca, 'GridAlpha', 0.3);

exportgraphics(gcf, fullfile(outDir, '02_overall_metrics.png'), ...
    'Resolution', 200);

% ==================================================================
% 图3: 逐类 F1 对比
% ==================================================================
fprintf('[3/6] 逐类F1...\n');
figure('Position', [100, 100, 1100, 650], 'Color', 'w');

rnF1 = data.rn_f1_per_class;
meF1 = data.me_f1_per_class;
y = 1:length(classNames);
h = 0.32;

b1h = barh(y - h/2, rnF1, h, 'FaceColor', c1, 'EdgeColor', 'w', ...
    'LineWidth', 1, 'FaceAlpha', 0.9);
hold on;
b2h = barh(y + h/2, meF1, h, 'FaceColor', c2, 'EdgeColor', 'w', ...
    'LineWidth', 1, 'FaceAlpha', 0.9);

% 数值标注
for i = 1:length(y)
    text(max(rnF1(i),0.01)+0.01, y(i)-h/2, sprintf('%.3f', rnF1(i)), ...
        'VerticalAlignment', 'middle', 'FontSize', 9, 'FontWeight', 'bold', ...
        'Color', c1);
    text(max(meF1(i),0.01)+0.01, y(i)+h/2, sprintf('%.3f', meF1(i)), ...
        'VerticalAlignment', 'middle', 'FontSize', 9, 'FontWeight', 'bold', ...
        'Color', c2);
end

set(gca, 'YTick', y, 'YTickLabel', classNames, 'FontSize', 11);
xlabel('F1-Score', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0, max([rnF1, meF1]) * 1.35]);
title('ResNet50 vs MoE — 逐类 F1 对比', 'FontSize', 14, 'FontWeight', 'bold');
legend([b1h, b2h], {'ResNet50 (后融合)', 'MoE (门控融合)'}, ...
    'Location', 'southeast', 'FontSize', 11);
grid on; set(gca, 'GridAlpha', 0.3, 'YDir', 'reverse');

exportgraphics(gcf, fullfile(outDir, '03_per_class_f1.png'), ...
    'Resolution', 200);

% ==================================================================
% 图4: 逐类 AUC 对比
% ==================================================================
fprintf('[4/6] 逐类AUC...\n');
figure('Position', [100, 100, 1100, 650], 'Color', 'w');

rnAUC = data.rn_auc_per_class;
meAUC = data.me_auc_per_class;

barh(y - h/2, rnAUC, h, 'FaceColor', c1, 'EdgeColor', 'w', ...
    'LineWidth', 1, 'FaceAlpha', 0.9);
hold on;
barh(y + h/2, meAUC, h, 'FaceColor', c2, 'EdgeColor', 'w', ...
    'LineWidth', 1, 'FaceAlpha', 0.9);

for i = 1:length(y)
    text(max(rnAUC(i),0.01)+0.01, y(i)-h/2, sprintf('%.4f', rnAUC(i)), ...
        'VerticalAlignment', 'middle', 'FontSize', 9, 'FontWeight', 'bold', ...
        'Color', c1);
    text(max(meAUC(i),0.01)+0.01, y(i)+h/2, sprintf('%.4f', meAUC(i)), ...
        'VerticalAlignment', 'middle', 'FontSize', 9, 'FontWeight', 'bold', ...
        'Color', c2);
end

set(gca, 'YTick', y, 'YTickLabel', classNames, 'FontSize', 11);
xlabel('AUC', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0, max([rnAUC, meAUC]) * 1.3]);
title('ResNet50 vs MoE — 逐类 AUC 对比', 'FontSize', 14, 'FontWeight', 'bold');
legend({'ResNet50 (后融合)', 'MoE (门控融合)'}, ...
    'Location', 'southeast', 'FontSize', 11);
grid on; set(gca, 'GridAlpha', 0.3, 'YDir', 'reverse');

exportgraphics(gcf, fullfile(outDir, '04_per_class_auc.png'), ...
    'Resolution', 200);

% ==================================================================
% 图5: ROC 曲线
% ==================================================================
fprintf('[5/6] ROC曲线...\n');
figure('Position', [50, 50, 1600, 1000], 'Color', 'w');

% 微平均 ROC (占据左侧大图)
subplot(3, 4, [1,2,5,6]);
plot(data.fpr_rn_micro, data.tpr_rn_micro, '-', 'Color', c1, ...
    'LineWidth', 2.5);
hold on;
plot(data.fpr_me_micro, data.tpr_me_micro, '-', 'Color', c2, ...
    'LineWidth', 2.5);
plot([0 1], [0 1], 'k--', 'LineWidth', 1.2, 'Color', [0.5 0.5 0.5]);
xlabel('False Positive Rate', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('True Positive Rate', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('ROC 曲线 (Micro-average)\nResNet50 AUC=%.4f  |  MoE AUC=%.4f', ...
    data.auc_rn_micro, data.auc_me_micro), 'FontSize', 13, 'FontWeight', 'bold');
legend({sprintf('ResNet50 (AUC=%.4f)', data.auc_rn_micro), ...
    sprintf('MoE (AUC=%.4f)', data.auc_me_micro), '随机猜测'}, ...
    'Location', 'southeast', 'FontSize', 11);
grid on; axis square; xlim([-0.02 1.02]); ylim([-0.02 1.02]);
set(gca, 'FontSize', 10);

% 逐类 ROC (8个子图)
for i = 1:8
    subplot(3, 4, i+4);
    fpr_rn_c = data.(sprintf('fpr_rn_%d', i-1));
    tpr_rn_c = data.(sprintf('tpr_rn_%d', i-1));
    fpr_me_c = data.(sprintf('fpr_me_%d', i-1));
    tpr_me_c = data.(sprintf('tpr_me_%d', i-1));

    plot(fpr_rn_c, tpr_rn_c, '-', 'Color', c1, 'LineWidth', 1.8);
    hold on;
    plot(fpr_me_c, tpr_me_c, '-', 'Color', c2, 'LineWidth', 1.8);
    plot([0 1], [0 1], 'k--', 'LineWidth', 0.8, 'Color', [0.6 0.6 0.6]);

    title(sprintf('%s', classNames{i}), 'FontSize', 12, 'FontWeight', 'bold');
    legend({sprintf('ResNet50 (%.3f)', data.rn_auc_per_class(i)), ...
        sprintf('MoE (%.3f)', data.me_auc_per_class(i))}, ...
        'Location', 'southeast', 'FontSize', 7.5);
    grid on; axis square;
    xlim([-0.02 1.02]); ylim([-0.02 1.02]);
    set(gca, 'FontSize', 9);
end

sgtitle('ResNet50 vs MoE — ROC 曲线分析', ...
    'FontSize', 16, 'FontWeight', 'bold');
exportgraphics(gcf, fullfile(outDir, '05_roc_curves.png'), ...
    'Resolution', 200);

% ==================================================================
% 图6: 推理速度 + 参数量对比
% ==================================================================
fprintf('[6/6] 推理速度+参数量...\n');
figure('Position', [100, 100, 1100, 500], 'Color', 'w');

% 推理速度
subplot(1,2,1);
times = [data.rn_inference_time_ms, data.me_inference_time_ms];
b = bar(times, 'FaceColor', 'flat', 'EdgeColor', 'w', 'LineWidth', 1.5, ...
    'BarWidth', 0.6);
b.CData(1,:) = c1;
b.CData(2,:) = c2;
set(gca, 'XTickLabel', {'ResNet50', 'MoE'});
for i = 1:2
    text(i, times(i) + 8, sprintf('%.1f ms', times(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold');
end
ylabel('推理时间 (ms/对)', 'FontSize', 12, 'FontWeight', 'bold');
title('单对眼底图推理速度对比', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0, max(times) * 1.2]);
grid on; set(gca, 'FontSize', 11);

% 模型参数量
subplot(1,2,2);
params = [data.rn_params, data.me_params, data.me_trainable_params] / 1e6;
b2 = bar(params, 'FaceColor', 'flat', 'EdgeColor', 'w', 'LineWidth', 1.5, ...
    'BarWidth', 0.6);
b2.CData(1,:) = c1;
b2.CData(2,:) = c2;
b2.CData(3,:) = [1.0 0.67 0.48];
set(gca, 'XTickLabel', {'ResNet50', 'MoE (总参数)', 'MoE (可训练)'});
for i = 1:3
    text(i, params(i) + 1.5, sprintf('%.1fM', params(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
end
ylabel('参数量 (百万)', 'FontSize', 12, 'FontWeight', 'bold');
title('模型参数量对比', 'FontSize', 14, 'FontWeight', 'bold');
ylim([0, max(params) * 1.2]);
grid on; set(gca, 'FontSize', 11);

sgtitle('ResNet50 vs MoE — 推理效率对比', ...
    'FontSize', 16, 'FontWeight', 'bold');
exportgraphics(gcf, fullfile(outDir, '06_model_comparison.png'), ...
    'Resolution', 200);

% ==================== 控制台输出 ====================
fprintf('\n========================================\n');
fprintf('MATLAB 对比实验图表生成完成！\n');
fprintf('========================================\n');
fprintf('ResNet50: Macro F1=%.4f | AUC=%.4f | 推理=%.1fms | 参数=%.1fM\n', ...
    data.rn_macro_f1, data.rn_macro_auc, ...
    data.rn_inference_time_ms, data.rn_params/1e6);
fprintf('MoE:      Macro F1=%.4f | AUC=%.4f | 推理=%.1fms | 参数=%.1fM(总)/%.1fM(可训)\n', ...
    data.me_macro_f1, data.me_macro_auc, ...
    data.me_inference_time_ms, data.me_params/1e6, data.me_trainable_params/1e6);
fprintf('输出目录: %s\n', outDir);

% 逐类 F1 对比表
fprintf('\n逐类 F1 对比:\n');
fprintf('%-12s %10s %10s %10s\n', '类别', 'ResNet50', 'MoE', '差异');
fprintf('%-12s %10s %10s %10s\n', '----------', '----------', '----------', '----------');
for i = 1:8
    diff = data.me_f1_per_class(i) - data.rn_f1_per_class(i);
    fprintf('%-12s %10.4f %10.4f %+10.4f\n', classNames{i}, ...
        data.rn_f1_per_class(i), data.me_f1_per_class(i), diff);
end
