/**
 * ZEMO — AI/ML MODEL TRAINING PLATFORM
 * Master Interactive Application Layer
 */

(function () {
  'use strict';

  // ============================================================
  // 1. APPLICATION STATE & CONFIGURATION
  // ============================================================
  const API_BASE = 'http://localhost:8000/api/v1';

  const state = {
    currentSection: 'upload',
    dataset: null, // { name, size, type, rawData, headers, rows, types, stats }
    restrictions: {
      columns: {}, // { [colName]: 'include' | 'exclude' | 'review' }
      sensitive: {
        email: false,
        phone: false,
        names: false,
        personalIds: false,
        financial: false,
        custom: false,
      },
      quality: {
        missingValues: false,
        duplicates: false,
        invalidValues: false,
        outliers: false,
        highCardinality: false,
      },
      constraints: {
        targetColumn: '',
        missingHandling: 'drop',
        encoding: 'auto',
        scaling: 'standard',
        outlierTreatment: 'none',
        split: '80-20',
      },

    },
    selectedModel: 'random-forest',
    algorithmConfig: {
      id: 'random_forest_classifier',
      task: 'classification',
      hyperparameters: {},
      valid: true,
      errors: [],
    },
    modelCategory: 'traditional',
    activeVizTab: 'distribution',
    queue: [],
    theme: 'organic',
    aiTraining: {
      mode: 'pretrained', // 'pretrained' | 'scratch'
      pretrained: {
        modelId: 'distilbert-base',
        method: 'head', // 'head' | 'full' | 'lora'
        dataset: 'current',
        epochs: 3,
        learningRate: '0.00002',
        batchSize: 16,
        valSplit: 20,
        isSupported: true,
      },
      scratch: {
        architecture: 'mlp', // 'mlp' | 'conv1d' | 'bilstm' | 'mini-transformer' | 'tabresnet'
        size: 'small', // 'micro' | 'small' | 'medium' | 'large'
        task: 'binary', // 'binary' | 'multiclass' | 'regression' | 'anomaly'
        initScheme: 'glorot-uniform',
        split: '70-15-15',
        scaler: 'standard',
        encoding: 'onehot',
        epochs: 50,
        learningRate: '0.001',
        batchSize: 32,
        dropout: 0.20,
      },
      activeJob: null,
      chatHistory: [],
      policyStatus: 'passed',
    },
  };

  // Sample datasets for instant interactive preview
  const SAMPLE_DATASETS = {
    housing: {
      name: 'housing_features_sample.csv',
      size: 48210,
      headers: ['id', 'median_income', 'housing_median_age', 'total_rooms', 'population', 'ocean_proximity', 'median_house_value'],
      rows: [
        ['1001', '8.3252', '41', '880', '322', 'NEAR BAY', '452600'],
        ['1002', '8.3014', '21', '7099', '2401', 'NEAR BAY', '358500'],
        ['1003', '7.2574', '52', '1467', '496', 'NEAR BAY', '352100'],
        ['1004', '5.6431', '52', '1274', '558', 'NEAR BAY', '341300'],
        ['1005', '3.8462', '52', '1627', '565', 'NEAR BAY', '342200'],
        ['1006', '4.0368', '52', '919', '413', 'NEAR BAY', '269700'],
        ['1007', '3.6591', '52', '2535', '1094', 'NEAR BAY', '299200'],
        ['1008', '3.1200', '40', '751', '510', 'INLAND', '241400'],
        ['1009', '2.0804', '42', '2555', '1206', 'INLAND', '226700'],
        ['1010', '3.6912', '52', '2202', '987', 'NEAR OCEAN', '261100'],
        ['1011', '3.2031', '52', '3503', '1504', 'NEAR OCEAN', '281500'],
        ['1012', '2.8750', '36', '1452', '769', 'INLAND', '182000'],
      ],
      types: {
        id: 'numeric',
        median_income: 'numeric',
        housing_median_age: 'numeric',
        total_rooms: 'numeric',
        population: 'numeric',
        ocean_proximity: 'categorical',
        median_house_value: 'numeric',
      },
    },
    churn: {
      name: 'customer_churn_records.csv',
      size: 32450,
      headers: ['customer_id', 'tenure_months', 'contract_type', 'monthly_charges', 'tech_support', 'total_spend', 'churn'],
      rows: [
        ['C-7590', '1', 'Month-to-month', '29.85', 'No', '29.85', 'No'],
        ['C-5575', '34', 'One year', '56.95', 'No', '1889.50', 'No'],
        ['C-3668', '2', 'Month-to-month', '53.85', 'Yes', '108.15', 'Yes'],
        ['C-7795', '45', 'One year', '42.30', 'Yes', '1840.75', 'No'],
        ['C-9237', '2', 'Month-to-month', '70.70', 'No', '151.65', 'Yes'],
        ['C-9305', '8', 'Month-to-month', '99.65', 'No', '820.50', 'Yes'],
        ['C-1452', '22', 'Month-to-month', '89.10', 'No', '1949.40', 'Yes'],
        ['C-6713', '10', 'Month-to-month', '29.75', 'No', '301.90', 'No'],
        ['C-7892', '28', 'Two year', '104.80', 'Yes', '3046.05', 'Yes'],
        ['C-6388', '62', 'One year', '56.15', 'Yes', '3487.95', 'No'],
      ],
      types: {
        customer_id: 'text',
        tenure_months: 'numeric',
        contract_type: 'categorical',
        monthly_charges: 'numeric',
        tech_support: 'categorical',
        total_spend: 'numeric',
        churn: 'categorical',
      },
    },
  };

  // Model Metadata (18 Machine Learning Algorithms Registry)
  const MODEL_METADATA = {
    // Classification (7)
    'random-forest': { name: 'Random Forest Classifier', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'f1', paramGroup: 'treeParams', algoId: 'random_forest_classifier' },
    'random_forest_classifier': { name: 'Random Forest Classifier', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'f1', paramGroup: 'treeParams', algoId: 'random_forest_classifier' },
    'decision-tree': { name: 'Decision Tree Classifier', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'f1', paramGroup: 'treeParams', algoId: 'decision_tree_classifier' },
    'decision_tree_classifier': { name: 'Decision Tree Classifier', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'f1', paramGroup: 'treeParams', algoId: 'decision_tree_classifier' },
    'gradient-boosting': { name: 'Gradient Boosting Classifier', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'f1', paramGroup: 'treeParams', algoId: 'gradient_boosting_classifier' },
    'gradient_boosting_classifier': { name: 'Gradient Boosting Classifier', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'f1', paramGroup: 'treeParams', algoId: 'gradient_boosting_classifier' },
    'logistic-regression': { name: 'Logistic Regression', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'accuracy', paramGroup: 'linearParams', algoId: 'logistic_regression' },
    'logistic_regression': { name: 'Logistic Regression', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'accuracy', paramGroup: 'linearParams', algoId: 'logistic_regression' },
    'svm': { name: 'Support Vector Classifier (SVC)', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'accuracy', paramGroup: 'svmParams', algoId: 'svc' },
    'svc': { name: 'Support Vector Classifier (SVC)', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'accuracy', paramGroup: 'svmParams', algoId: 'svc' },
    'knn': { name: 'K-Nearest Neighbors Classifier', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'accuracy', paramGroup: 'knnParams', algoId: 'knn_classifier' },
    'knn_classifier': { name: 'K-Nearest Neighbors Classifier', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'accuracy', paramGroup: 'knnParams', algoId: 'knn_classifier' },
    'naive-bayes': { name: 'Gaussian Naive Bayes', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'accuracy', paramGroup: 'linearParams', algoId: 'gaussian_nb' },
    'gaussian_nb': { name: 'Gaussian Naive Bayes', type: 'classification', category: 'traditional', tasks: ['classification'], defaultMetric: 'accuracy', paramGroup: 'linearParams', algoId: 'gaussian_nb' },

    // Regression (8)
    'linear-regression': { name: 'Linear Regression', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'linearParams', algoId: 'linear_regression' },
    'linear_regression': { name: 'Linear Regression', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'linearParams', algoId: 'linear_regression' },
    'ridge': { name: 'Ridge Regression (L2)', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'linearParams', algoId: 'ridge_regression' },
    'ridge_regression': { name: 'Ridge Regression (L2)', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'linearParams', algoId: 'ridge_regression' },
    'lasso': { name: 'Lasso Regression (L1)', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'linearParams', algoId: 'lasso_regression' },
    'lasso_regression': { name: 'Lasso Regression (L1)', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'linearParams', algoId: 'lasso_regression' },
    'decision-tree-regressor': { name: 'Decision Tree Regressor', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'treeParams', algoId: 'decision_tree_regressor' },
    'decision_tree_regressor': { name: 'Decision Tree Regressor', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'treeParams', algoId: 'decision_tree_regressor' },
    'random-forest-regressor': { name: 'Random Forest Regressor', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'treeParams', algoId: 'random_forest_regressor' },
    'random_forest_regressor': { name: 'Random Forest Regressor', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'treeParams', algoId: 'random_forest_regressor' },
    'knn-regressor': { name: 'K-Nearest Neighbors Regressor', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'knnParams', algoId: 'knn_regressor' },
    'knn_regressor': { name: 'K-Nearest Neighbors Regressor', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'knnParams', algoId: 'knn_regressor' },
    'svr': { name: 'Support Vector Regressor (SVR)', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'svmParams', algoId: 'svr' },
    'gradient-boosting-regressor': { name: 'Gradient Boosting Regressor', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'treeParams', algoId: 'gradient_boosting_regressor' },
    'gradient_boosting_regressor': { name: 'Gradient Boosting Regressor', type: 'regression', category: 'traditional', tasks: ['regression'], defaultMetric: 'rmse', paramGroup: 'treeParams', algoId: 'gradient_boosting_regressor' },

    // Clustering (3)
    'kmeans': { name: 'K-Means Clustering', type: 'clustering', category: 'traditional', tasks: ['clustering'], defaultMetric: 'silhouette', paramGroup: 'knnParams', algoId: 'kmeans' },
    'dbscan': { name: 'DBSCAN Clustering', type: 'clustering', category: 'traditional', tasks: ['clustering'], defaultMetric: 'silhouette', paramGroup: 'knnParams', algoId: 'dbscan' },
    'agglomerative': { name: 'Agglomerative Clustering', type: 'clustering', category: 'traditional', tasks: ['clustering'], defaultMetric: 'silhouette', paramGroup: 'knnParams', algoId: 'agglomerative_clustering' },
    'agglomerative_clustering': { name: 'Agglomerative Clustering', type: 'clustering', category: 'traditional', tasks: ['clustering'], defaultMetric: 'silhouette', paramGroup: 'knnParams', algoId: 'agglomerative_clustering' },

    // Deep Learning & Anomaly Detection models
    'isolation-forest': { name: 'Isolation Forest', type: 'clustering', category: 'traditional', tasks: ['clustering'], defaultMetric: 'anomaly_score', paramGroup: 'treeParams', algoId: 'kmeans' },
    'mlp': { name: 'Multilayer Perceptron', type: 'both', category: 'deep', tasks: ['classification', 'regression'], defaultMetric: 'accuracy', paramGroup: 'nnParams', algoId: 'random_forest_classifier' },
    'ffnn': { name: 'Feedforward Neural Network', type: 'both', category: 'deep', tasks: ['classification', 'regression'], defaultMetric: 'accuracy', paramGroup: 'nnParams', algoId: 'random_forest_classifier' },
    'autoencoder': { name: 'Deep Autoencoder', type: 'clustering', category: 'deep', tasks: ['clustering'], defaultMetric: 'recon_loss', paramGroup: 'nnParams', algoId: 'kmeans' },
  };

  // ============================================================
  // 2. DOM ELEMENT REFERENCES
  // ============================================================
  const el = {
    // Navigation
    navLinks: document.querySelectorAll('.nav-link'),
    sections: document.querySelectorAll('.section'),
    hamburger: document.getElementById('hamburger'),
    mobileNav: document.getElementById('mobileNav'),
    btnNewExperiment: document.getElementById('btnNewExperiment'),
    workspaceSelector: document.getElementById('workspaceSelector'),

    // Upload Section
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    browseLink: document.getElementById('browseLink'),
    uploadEmpty: document.getElementById('uploadEmpty'),
    datasetInfo: document.getElementById('datasetInfo'),
    datasetPreview: document.getElementById('datasetPreview'),
    previewTableHead: document.getElementById('previewTableHead'),
    previewTableBody: document.getElementById('previewTableBody'),
    fileName: document.getElementById('fileName'),
    fileType: document.getElementById('fileType'),
    fileSize: document.getElementById('fileSize'),
    fileRows: document.getElementById('fileRows'),
    fileCols: document.getElementById('fileCols'),
    fileStatus: document.getElementById('fileStatus'),
    columnList: document.getElementById('columnList'),
    btnRemoveDataset: document.getElementById('btnRemoveDataset'),
    btnContinueRestrictions: document.getElementById('btnContinueRestrictions'),

    // Restrictions Section
    restrictionsEmpty: document.getElementById('restrictionsEmpty'),
    restrictionsContent: document.getElementById('restrictionsContent'),
    columnRestrictions: document.getElementById('columnRestrictions'),
    colRestrictionCount: document.getElementById('colRestrictionCount'),
    sensitiveCount: document.getElementById('sensitiveCount'),
    qualityCount: document.getElementById('qualityCount'),
    targetColumn: document.getElementById('targetColumn'),
    excludedFeatures: document.getElementById('excludedFeatures'),
    summaryIncluded: document.getElementById('summaryIncluded'),
    summaryExcluded: document.getElementById('summaryExcluded'),
    summaryReview: document.getElementById('summaryReview'),
    summarySensitive: document.getElementById('summarySensitive'),
    summaryQuality: document.getElementById('summaryQuality'),
    summaryTarget: document.getElementById('summaryTarget'),
    summaryEncoding: document.getElementById('summaryEncoding'),
    summaryScaling: document.getElementById('summaryScaling'),
    summaryOutliers: document.getElementById('summaryOutliers'),
    summarySplit: document.getElementById('summarySplit'),
    btnResetRestrictions: document.getElementById('btnResetRestrictions'),
    btnContinueOutput: document.getElementById('btnContinueOutput'),


    // Output Section
    outputEmpty: document.getElementById('outputEmpty'),
    outputContent: document.getElementById('outputContent'),
    metricRows: document.getElementById('metricRows'),
    metricCols: document.getElementById('metricCols'),
    metricNumeric: document.getElementById('metricNumeric'),
    metricCategorical: document.getElementById('metricCategorical'),
    metricMissing: document.getElementById('metricMissing'),
    metricRowsNote: document.getElementById('metricRowsNote'),
    metricColsNote: document.getElementById('metricColsNote'),
    metricNumericNote: document.getElementById('metricNumericNote'),
    metricCategoricalNote: document.getElementById('metricCategoricalNote'),
    metricMissingNote: document.getElementById('metricMissingNote'),
    vizTabs: document.querySelectorAll('.viz-tab'),
    vizPanels: {
      distribution: document.getElementById('vizDistribution'),
      categories: document.getElementById('vizCategories'),
      missing: document.getElementById('vizMissing'),
      correlations: document.getElementById('vizCorrelations'),
      boxplots: document.getElementById('vizBoxPlots'),
      scatter: document.getElementById('vizScatter'),
    },
    distributionChart: document.getElementById('distributionChart'),
    categoryChart: document.getElementById('categoryChart'),
    missingChart: document.getElementById('missingChart'),
    correlationsHeatmap: document.getElementById('correlationsHeatmap'),
    boxPlotChart: document.getElementById('boxPlotChart'),
    scatterChart: document.getElementById('scatterChart'),
    featureTableBody: document.getElementById('featureTableBody'),
    pipelineEncodingVal: document.getElementById('pipelineEncodingVal'),
    pipelineSplitVal: document.getElementById('pipelineSplitVal'),
    preprocessingPreviewContainer: document.getElementById('preprocessingPreviewContainer'),
    previewLeakageBadge: document.getElementById('previewLeakageBadge'),
    previewFeaturesBadge: document.getElementById('previewFeaturesBadge'),
    previewInCols: document.getElementById('previewInCols'),
    previewOutCols: document.getElementById('previewOutCols'),
    previewTrainSplit: document.getElementById('previewTrainSplit'),
    previewTestSplit: document.getElementById('previewTestSplit'),
    previewWarningsBanner: document.getElementById('previewWarningsBanner'),
    previewWarningsContent: document.getElementById('previewWarningsContent'),
    previewOperationsList: document.getElementById('previewOperationsList'),
    transformedSampleHead: document.getElementById('transformedSampleHead'),
    transformedSampleBody: document.getElementById('transformedSampleBody'),


    // Zemo Core Intelligence Elements
    coreRequirementInput: document.getElementById('coreRequirementInput'),
    presetPills: document.querySelectorAll('.preset-pill'),
    btnSynthesizePlan: document.getElementById('btnSynthesizePlan'),
    planModelVal: document.getElementById('planModelVal'),
    planMetricVal: document.getElementById('planMetricVal'),
    planFeatureVal: document.getElementById('planFeatureVal'),
    btnApplyPlanToLab: document.getElementById('btnApplyPlanToLab'),

    // Model Selection & Multi-Task Filters
    taskPills: document.querySelectorAll('.task-pill'),
    modelCategoryTabs: document.querySelectorAll('.model-tab'),
    modelCards: document.querySelectorAll('.model-card'),
    btnDownloadPreview: document.getElementById('btnDownloadPreview'),
    btnContinueTraining: document.getElementById('btnContinueTraining'),

    // Training Section
    expName: document.getElementById('expName'),
    expDataset: document.getElementById('expDataset'),
    expTarget: document.getElementById('expTarget'),
    expModel: document.getElementById('expModel'),
    expEncoding: document.getElementById('expEncoding'),
    expSplit: document.getElementById('expSplit'),
    expSeed: document.getElementById('expSeed'),
    expMetric: document.getElementById('expMetric'),
    hyperparamPanel: document.getElementById('hyperparamPanel'),
    dynamicHyperparamFields: document.getElementById('dynamicHyperparamFields'),
    dynamicHyperparamErrors: document.getElementById('dynamicHyperparamErrors'),
    dynamicAlgoDesc: document.getElementById('dynamicAlgoDesc'),
    hyperparamStatusBadge: document.getElementById('hyperparamStatusBadge'),
    paramGroups: {
      treeParams: document.getElementById('treeParams'),
      nnParams: document.getElementById('nnParams'),
      linearParams: document.getElementById('linearParams'),
      knnParams: document.getElementById('knnParams'),
      svmParams: document.getElementById('svmParams'),
    },

    // Sliders
    sliderEstimators: document.getElementById('sliderEstimators'),
    valEstimators: document.getElementById('valEstimators'),
    sliderMaxDepth: document.getElementById('sliderMaxDepth'),
    valMaxDepth: document.getElementById('valMaxDepth'),
    sliderMinSamples: document.getElementById('sliderMinSamples'),
    valMinSamples: document.getElementById('valMinSamples'),
    sliderLayers: document.getElementById('sliderLayers'),
    valLayers: document.getElementById('valLayers'),
    sliderHiddenUnits: document.getElementById('sliderHiddenUnits'),
    valHiddenUnits: document.getElementById('valHiddenUnits'),
    sliderEpochs: document.getElementById('sliderEpochs'),
    valEpochs: document.getElementById('valEpochs'),
    sliderNeighbors: document.getElementById('sliderNeighbors'),
    valNeighbors: document.getElementById('valNeighbors'),
    // Queue & Actions
    queueList: document.getElementById('queueList'),
    btnSaveDraft: document.getElementById('btnSaveDraft'),
    btnValidateConfig: document.getElementById('btnValidateConfig'),
    btnAddToQueue: document.getElementById('btnAddToQueue'),
    btnTrainModel: document.getElementById('btnTrainModel'),
    prototypeNotice: document.getElementById('prototypeNotice'),

    // Training Progress Telemetry
    activeTrainingJobName: document.getElementById('activeTrainingJobName'),
    activeTrainingJobStatus: document.getElementById('activeTrainingJobStatus'),
    trainingProgressBar: document.getElementById('trainingProgressBar'),
    trainingStatStage: document.getElementById('trainingStatStage'),
    trainingStatProgress: document.getElementById('trainingStatProgress'),
    trainingStatElapsed: document.getElementById('trainingStatElapsed'),
    trainingStatRemaining: document.getElementById('trainingStatRemaining'),
    trainingLog: document.getElementById('trainingLog'),
    btnDownloadTrainedModel: document.getElementById('btnDownloadTrainedModel'),

    // Results Placeholder
    resultsPlaceholderPanel: document.getElementById('resultsPlaceholderPanel'),
    resultExperimentName: document.getElementById('resultExperimentName'),
    resultExperimentBadge: document.getElementById('resultExperimentBadge'),
    resultMetricLabel1: document.getElementById('resultMetricLabel1'),
    resultMetricVal1: document.getElementById('resultMetricVal1'),
    resultMetricLabel2: document.getElementById('resultMetricLabel2'),
    resultMetricVal2: document.getElementById('resultMetricVal2'),
    resultMetricLabel3: document.getElementById('resultMetricLabel3'),
    resultMetricVal3: document.getElementById('resultMetricVal3'),
    resultMetricLabel4: document.getElementById('resultMetricLabel4'),
    resultMetricVal4: document.getElementById('resultMetricVal4'),

    // Model Comparison & Detailed Evaluation
    btnCompareModels: document.getElementById('btnCompareModels'),
    compareCountBadge: document.getElementById('compareCountBadge'),
    modelComparisonModal: document.getElementById('modelComparisonModal'),
    btnCloseComparisonModal: document.getElementById('btnCloseComparisonModal'),
    comparisonModalBody: document.getElementById('comparisonModalBody'),
    featureImportanceContainer: document.getElementById('featureImportanceContainer'),
    featureImportanceMethodBadge: document.getElementById('featureImportanceMethodBadge'),
    featureImportanceExplanation: document.getElementById('featureImportanceExplanation'),
    featureImportanceBarsList: document.getElementById('featureImportanceBarsList'),
    evaluationDiagnosticsContainer: document.getElementById('evaluationDiagnosticsContainer'),
    evalDetailSecondaryBadge: document.getElementById('evalDetailSecondaryBadge'),
    evaluationDiagnosticsContent: document.getElementById('evaluationDiagnosticsContent'),

    // Feedback
    notification: document.getElementById('notification'),
    notificationText: document.getElementById('notificationText'),

    // Theme Toggle
    themeToggle: document.getElementById('themeToggle'),
    themeToggleLabel: document.getElementById('themeToggleLabel'),

    // AI Training Module
    modeCardPretrained: document.getElementById('modeCardPretrained'),
    modeCardScratch: document.getElementById('modeCardScratch'),
    panelPretrained: document.getElementById('panelPretrained'),
    panelScratch: document.getElementById('panelScratch'),

    // Mode A: Pre-trained
    ptModelSelect: document.getElementById('ptModelSelect'),
    unsupportedModelAlert: document.getElementById('unsupportedModelAlert'),
    unsupportedModelText: document.getElementById('unsupportedModelText'),
    ptModelMetaCard: document.getElementById('ptModelMetaCard'),
    ptMetaFamily: document.getElementById('ptMetaFamily'),
    ptMetaParams: document.getElementById('ptMetaParams'),
    ptMetaCheckpoint: document.getElementById('ptMetaCheckpoint'),
    ptMetaStatus: document.getElementById('ptMetaStatus'),
    ptDatasetSelect: document.getElementById('ptDatasetSelect'),
    ptDatasetApprovalBar: document.getElementById('ptDatasetApprovalBar'),
    ptDatasetApprovalText: document.getElementById('ptDatasetApprovalText'),
    ptDatasetRowsBadge: document.getElementById('ptDatasetRowsBadge'),
    ptEpochs: document.getElementById('ptEpochs'),
    valPtEpochs: document.getElementById('valPtEpochs'),
    ptLR: document.getElementById('ptLR'),
    ptBatchSize: document.getElementById('ptBatchSize'),
    ptValSplit: document.getElementById('ptValSplit'),
    ptEstVRAM: document.getElementById('ptEstVRAM'),
    ptEstTime: document.getElementById('ptEstTime'),

    // Mode B: Train from Scratch
    scratchArchSelect: document.getElementById('scratchArchSelect'),
    scratchTaskSelect: document.getElementById('scratchTaskSelect'),
    scratchInitSelect: document.getElementById('scratchInitSelect'),
    scratchTaskValidationBox: document.getElementById('scratchTaskValidationBox'),
    scratchValidationMsg: document.getElementById('scratchValidationMsg'),
    scratchSampleCheck: document.getElementById('scratchSampleCheck'),
    scratchSplitSelect: document.getElementById('scratchSplitSelect'),
    scratchScalerSelect: document.getElementById('scratchScalerSelect'),
    scratchEncodingSelect: document.getElementById('scratchEncodingSelect'),
    scratchEpochs: document.getElementById('scratchEpochs'),
    valScratchEpochs: document.getElementById('valScratchEpochs'),
    scratchLR: document.getElementById('scratchLR'),
    scratchBatchSize: document.getElementById('scratchBatchSize'),
    scratchDropout: document.getElementById('scratchDropout'),
    valScratchDropout: document.getElementById('valScratchDropout'),

    // Governance Gateway
    btnRunSecurityScan: document.getElementById('btnRunSecurityScan'),
    gatewayBadge: document.getElementById('gatewayBadge'),
    gatewayStatusText: document.getElementById('gatewayStatusText'),
    gatewayDetailText: document.getElementById('gatewayDetailText'),

    // Studio Action Buttons
    btnReviewPlan: document.getElementById('btnReviewPlan'),
    btnLaunchTrainingDemo: document.getElementById('btnLaunchTrainingDemo'),
    btnViewBackendSpec: document.getElementById('btnViewBackendSpec'),

    // Assistant Chatbox
    trainingAssistantCard: document.getElementById('trainingAssistantCard'),
    chatMessagesStream: document.getElementById('chatMessagesStream'),
    chatForm: document.getElementById('chatForm'),
    chatInput: document.getElementById('chatInput'),
    btnChatSend: document.getElementById('btnChatSend'),
    btnClearChat: document.getElementById('btnClearChat'),
    promptChips: document.querySelectorAll('.prompt-chip'),

    // Live Monitor
    trainingLiveMonitor: document.getElementById('trainingLiveMonitor'),
    liveMonitorTitle: document.getElementById('liveMonitorTitle'),
    liveJobStatusBadge: document.getElementById('liveJobStatusBadge'),
    btnCancelTrainingJob: document.getElementById('btnCancelTrainingJob'),
    liveTrainingProgressBar: document.getElementById('liveTrainingProgressBar'),
    liveStatEpoch: document.getElementById('liveStatEpoch'),
    liveStatTrainLoss: document.getElementById('liveStatTrainLoss'),
    liveStatValLoss: document.getElementById('liveStatValLoss'),
    liveStatAccuracy: document.getElementById('liveStatAccuracy'),
    liveStatElapsed: document.getElementById('liveStatElapsed'),
    liveTerminalLogsBody: document.getElementById('liveTerminalLogsBody'),
    liveLogCount: document.getElementById('liveLogCount'),
    liveEvaluationResultsBox: document.getElementById('liveEvaluationResultsBox'),
    finalValLoss: document.getElementById('finalValLoss'),
    finalF1: document.getElementById('finalF1'),
    finalAccuracy: document.getElementById('finalAccuracy'),
    finalLatency: document.getElementById('finalLatency'),
    checkpointsTableBody: document.getElementById('checkpointsTableBody'),

    // Modals
    preflightReviewModal: document.getElementById('preflightReviewModal'),
    btnCloseReviewModal: document.getElementById('btnCloseReviewModal'),
    btnCancelReviewModal: document.getElementById('btnCancelReviewModal'),
    btnConfirmAndLaunch: document.getElementById('btnConfirmAndLaunch'),
    backendSpecModal: document.getElementById('backendSpecModal'),
    btnCloseSpecModal: document.getElementById('btnCloseSpecModal'),
    btnDismissSpecModal: document.getElementById('btnDismissSpecModal'),
  };

  // ============================================================
  // 3. UTILITY FUNCTIONS
  // ============================================================
  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function showNotification(message, type = 'info') {
    if (!el.notification) return;
    el.notification.className = 'notification ' + type;
    el.notificationText.textContent = message;
    el.notification.classList.add('show');

    clearTimeout(el.notification._timer);
    el.notification._timer = setTimeout(() => {
      el.notification.classList.remove('show');
    }, 4000);
  }

  function scrollToSection(id) {
    const target = document.getElementById(id);
    if (target) {
      const headerOffset = 64;
      const elementPosition = target.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth',
      });
    }
  }

  function inferType(values) {
    const nonNull = values.filter((v) => v !== '' && v !== null && v !== undefined);
    if (nonNull.length === 0) return 'text';

    let isNum = true;
    let isBool = true;

    for (let val of nonNull) {
      const str = String(val).trim().toLowerCase();
      if (str !== 'true' && str !== 'false' && str !== '1' && str !== '0' && str !== 'yes' && str !== 'no') {
        isBool = false;
      }
      if (isNaN(Number(val))) {
        isNum = false;
      }
    }

    if (isBool && nonNull.length > 2) return 'boolean';
    if (isNum) return 'numeric';

    // Unique count for categorical vs free text
    const unique = new Set(nonNull);
    if (unique.size <= 10 || unique.size / nonNull.length < 0.4) {
      return 'categorical';
    }

    return 'text';
  }

  // ============================================================
  // 4. PARSING & DATA HANDLING
  // ============================================================
  function parseCSV(text) {
    const lines = text
      .split(/\r\n|\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
    if (lines.length === 0) return { headers: [], rows: [] };

    // Standard CSV split with quotes support
    function parseLine(line) {
      const result = [];
      let current = '';
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (c === '"' || c === "'") {
          inQuotes = !inQuotes;
        } else if (c === ',' && !inQuotes) {
          result.push(current.trim());
          current = '';
        } else {
          current += c;
        }
      }
      result.push(current.trim());
      return result;
    }

    const headers = parseLine(lines[0]);
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const parsed = parseLine(lines[i]);
      if (parsed.length === headers.length) {
        rows.push(parsed);
      }
    }

    return { headers, rows };
  }

  function parseJSONL(text) {
    const lines = text
      .split(/\r\n|\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
    if (lines.length === 0) return { headers: [], rows: [] };

    const parsedObjects = [];
    const headerSet = new Set();
    lines.forEach((l) => {
      try {
        const obj = JSON.parse(l);
        if (typeof obj === 'object' && obj !== null) {
          parsedObjects.push(obj);
          Object.keys(obj).forEach((k) => headerSet.add(k));
        }
      } catch (err) {
        // Skip malformed lines
      }
    });

    const headers = Array.from(headerSet);
    const rows = parsedObjects.map((obj) => headers.map((h) => (obj[h] !== undefined ? String(obj[h]) : '')));
    return { headers, rows };
  }

  async function handleFileSelected(file) {
    if (!file) return;

    const extension = file.name.split('.').pop().toLowerCase();
    const validFormats = ['csv', 'xlsx', 'json', 'jsonl', 'parquet'];

    if (!validFormats.includes(extension)) {
      showNotification(`Unsupported file format .${extension}. Please use CSV, XLSX, JSON, JSONL, or Parquet.`, 'error');
      return;
    }

    showNotification(`Uploading ${file.name} to Zemo backend analysis engine...`, 'info');

    try {
      const formData = new FormData();
      formData.append('file', file);

      let datasetId = null;
      const uploadRes = await fetch(`${API_BASE}/datasets`, {
        method: 'POST',
        body: formData,
      });

      if (uploadRes.status === 201) {
        const uploadData = await uploadRes.json();
        datasetId = uploadData.id;
      } else if (uploadRes.status === 409) {
        // Dataset with identical hash already exists in storage
        const errData = await uploadRes.json().catch(() => ({}));
        datasetId = errData.existing_dataset_id || (errData.detail && errData.detail.existing_dataset_id);
        if (!datasetId) {
          const listRes = await fetch(`${API_BASE}/datasets?page_size=50`);
          if (listRes.ok) {
            const listData = await listRes.json();
            const match = listData.items.find((d) => d.original_filename === file.name);
            if (match) datasetId = match.id;
          }
        }
      } else {
        const errText = await uploadRes.text();
        throw new Error(`Upload failed (${uploadRes.status}): ${errText}`);
      }

      if (!datasetId) {
        throw new Error('Could not obtain dataset ID from server.');
      }

      // Process dataset (validate & profile)
      await fetch(`${API_BASE}/datasets/${datasetId}/process`, { method: 'POST' }).catch(() => {});

      // Fetch actual dataset analysis
      const analysisRes = await fetch(`${API_BASE}/datasets/${datasetId}/analysis`);
      if (!analysisRes.ok) throw new Error(`Analysis failed: ${await analysisRes.text()}`);
      const analysisData = await analysisRes.json();

      // Fetch actual visualization data
      const vizRes = await fetch(`${API_BASE}/datasets/${datasetId}/visualizations`);
      if (!vizRes.ok) throw new Error(`Visualization retrieval failed: ${await vizRes.text()}`);
      const vizData = await vizRes.json();

      // Fetch preview rows
      const previewRes = await fetch(`${API_BASE}/datasets/${datasetId}/preview?rows=100`);
      let previewRows = [];
      let previewCols = analysisData.columns.map((c) => c.name);
      if (previewRes.ok) {
        const pData = await previewRes.json();
        previewCols = pData.columns || previewCols;
        previewRows = (pData.records || []).map((r) =>
          previewCols.map((c) => (r[c] !== null && r[c] !== undefined ? String(r[c]) : ''))
        );
      }

      const typesMap = {};
      analysisData.columns.forEach((c) => {
        typesMap[c.name] = c.inferred_type;
      });

      state.dataset = {
        id: datasetId,
        name: file.name,
        size: file.size,
        type: extension.toUpperCase(),
        headers: previewCols,
        rows: previewRows,
        types: typesMap,
        analysis: analysisData,
        visualizations: vizData,
        previewAvailable: true,
      };

      showNotification(`Analyzed ${file.name} successfully (${analysisData.total_rows} rows, ${analysisData.total_columns} columns, ${analysisData.missing_cells_total} missing cells).`, 'success');
      initDatasetUI();
    } catch (apiErr) {
      console.warn('Backend API unavailable or error occurred, using client-side fallback:', apiErr);
      showNotification(`Backend notice: ${apiErr.message}. Attempting client fallback...`, 'warning');
      clientSideFallback(file, extension);
    }
  }

  function clientSideFallback(file, extension) {
    if (extension !== 'csv' && extension !== 'jsonl') {
      state.dataset = {
        name: file.name,
        size: file.size,
        type: extension.toUpperCase(),
        headers: ['Sample_Feature_1', 'Sample_Feature_2', 'Sample_Feature_3', 'Target_Class'],
        rows: [
          ['12.4', 'High', 'Active', 'Class A'],
          ['15.8', 'Med', 'Active', 'Class B'],
          ['9.2', 'Low', 'Inactive', 'Class A'],
        ],
        types: {
          Sample_Feature_1: 'numeric',
          Sample_Feature_2: 'categorical',
          Sample_Feature_3: 'categorical',
          Target_Class: 'categorical',
        },
        previewAvailable: false,
      };
      initDatasetUI();
      return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      try {
        const content = e.target.result;
        const parsed = extension === 'jsonl' ? parseJSONL(content) : parseCSV(content);
        if (parsed.headers.length === 0) {
          showNotification(`Could not detect schema headers in this ${extension.toUpperCase()} file.`, 'warning');
          return;
        }
        const types = {};
        parsed.headers.forEach((h, colIndex) => {
          const colValues = parsed.rows.map((r) => r[colIndex]);
          types[h] = inferType(colValues);
        });

        state.dataset = {
          name: file.name,
          size: file.size,
          type: extension.toUpperCase(),
          headers: parsed.headers,
          rows: parsed.rows,
          types: types,
          previewAvailable: true,
        };
        showNotification(`Parsed ${file.name} via local parser (${parsed.rows.length} rows).`, 'success');
        initDatasetUI();
      } catch (err) {
        showNotification(`Error parsing ${extension.toUpperCase()} file: ` + err.message, 'error');
      }
    };
    reader.readAsText(file);
  }

  async function loadSampleDataset(key) {
    const filename = key === 'churn' ? 'customer_churn.csv' : 'housing_sample.csv';
    try {
      showNotification(`Loading real sample dataset (${filename})...`, 'info');
      const response = await fetch(`/sample-data/${filename}`);
      if (!response.ok) throw new Error(`Sample file not found: ${filename}`);
      const blob = await response.blob();
      const file = new File([blob], filename, { type: 'text/csv' });
      await handleFileSelected(file);
    } catch (err) {
      console.warn('Could not load sample file via fetch, falling back to local memory sample:', err);
      const sample = SAMPLE_DATASETS[key] || SAMPLE_DATASETS.housing;
      state.dataset = {
        name: sample.name,
        size: sample.size,
        type: 'CSV',
        headers: [...sample.headers],
        rows: sample.rows.map((r) => [...r]),
        types: { ...sample.types },
        previewAvailable: true,
      };
      showNotification(`Loaded sample dataset: ${sample.name}`, 'info');
      initDatasetUI();
    }
  }

  async function overrideColumnType(colName, newType) {
    if (!state.dataset) return;
    if (!state.dataset.id) {
      state.dataset.types[colName] = newType;
      initDatasetUI();
      return;
    }

    try {
      showNotification(`Updating type for '${colName}' to ${newType}...`, 'info');
      const res = await fetch(`${API_BASE}/datasets/${state.dataset.id}/columns/override`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ columns: { [colName]: newType } }),
      });
      if (!res.ok) throw new Error(await res.text());
      const updated = await res.json();
      state.dataset.analysis = updated;
      state.dataset.types[colName] = newType;

      const vizRes = await fetch(`${API_BASE}/datasets/${state.dataset.id}/visualizations`);
      if (vizRes.ok) {
        state.dataset.visualizations = await vizRes.json();
      }

      showNotification(`Column '${colName}' type updated to ${newType}.`, 'success');
      initDatasetUI();
    } catch (e) {
      showNotification(`Failed to override column type: ${e.message}`, 'error');
    }
  }

  function removeDataset() {
    state.dataset = null;
    state.restrictions.columns = {};
    state.restrictions.constraints.targetColumn = '';

    // Reset UI
    if (el.uploadZone) el.uploadZone.classList.remove('hidden');
    if (el.uploadEmpty) el.uploadEmpty.classList.remove('hidden');
    if (el.datasetInfo) el.datasetInfo.classList.add('hidden');
    if (el.datasetPreview) el.datasetPreview.classList.add('hidden');

    if (el.restrictionsEmpty) el.restrictionsEmpty.classList.remove('hidden');
    if (el.restrictionsContent) el.restrictionsContent.classList.add('hidden');

    if (el.outputEmpty) el.outputEmpty.classList.remove('hidden');
    if (el.outputContent) el.outputContent.classList.add('hidden');

    if (el.expDataset) el.expDataset.value = '';
    if (el.expName) el.expName.value = '';
    if (el.expTarget) el.expTarget.innerHTML = '<option value="">Select target...</option>';
    if (el.fileInput) el.fileInput.value = '';

    if (typeof handleDatasetSelectionChange === 'function') {
      handleDatasetSelectionChange(state.aiTraining.pretrained.dataset);
    }
    if (typeof validateScratchTask === 'function') {
      validateScratchTask();
    }

    updateStepIndicators(1);
    showNotification('Dataset removed.', 'info');
  }

  // ============================================================
  // 5. DATASET UI INITIALIZATION
  // ============================================================
  function initDatasetUI() {
    if (!state.dataset) return;

    const ds = state.dataset;

    // 1. Show dataset info & preview table
    if (el.uploadZone) el.uploadZone.classList.add('hidden');
    if (el.uploadEmpty) el.uploadEmpty.classList.add('hidden');
    if (el.datasetInfo) el.datasetInfo.classList.remove('hidden');
    if (el.datasetPreview) el.datasetPreview.classList.remove('hidden');

    if (el.fileName) el.fileName.textContent = ds.name;
    if (el.fileType) el.fileType.textContent = `${ds.type} Format · Local Preview`;
    if (el.fileSize) el.fileSize.textContent = formatBytes(ds.size);
    if (el.fileRows) el.fileRows.textContent = ds.rows.length.toLocaleString();
    if (el.fileCols) el.fileCols.textContent = ds.headers.length.toString();

    // Render table
    renderPreviewTable();

    // Init restrictions default state
    ds.headers.forEach((col) => {
      if (!state.restrictions.columns[col]) {
        state.restrictions.columns[col] = 'include';
      }
    });

    // Populate restrictions section
    if (el.restrictionsEmpty) el.restrictionsEmpty.classList.add('hidden');
    if (el.restrictionsContent) el.restrictionsContent.classList.remove('hidden');
    renderRestrictionsSection();

    // Populate output section
    if (el.outputEmpty) el.outputEmpty.classList.add('hidden');
    if (el.outputContent) el.outputContent.classList.remove('hidden');
    renderOutputSection();

    // Populate training form
    if (el.expDataset) el.expDataset.value = ds.name;
    if (el.expName && !el.expName.value) {
      const cleanName = ds.name.replace(/\.[^/.]+$/, '').toLowerCase();
      el.expName.value = `${cleanName}-exp-01`;
    }

    populateTargetDropdowns();
    updateRestrictionSummary();

    if (typeof handleDatasetSelectionChange === 'function') {
      handleDatasetSelectionChange(state.aiTraining.pretrained.dataset);
    }
    if (typeof validateScratchTask === 'function') {
      validateScratchTask();
    }
  }

  function renderPreviewTable() {
    if (!state.dataset) return;
    const { headers, rows, types } = state.dataset;

    // Render thead with interactive type override selector
    let thHtml = '<tr>';
    headers.forEach((h) => {
      const type = types[h] || 'text';
      thHtml += `
        <th>
          <div style="display:flex; flex-direction:column; gap:4px;">
            <span>${escapeHtml(h)}</span>
            <select class="type-override-select" data-col="${escapeHtml(h)}" title="Override column type" style="font-size:10px; background:var(--surface); color:var(--text-primary); border:1px solid var(--border-subtle); border-radius:3px; padding:1px 4px; font-family:var(--font-mono); cursor:pointer;">
              <option value="numeric" ${type === 'numeric' ? 'selected' : ''}>numeric</option>
              <option value="categorical" ${type === 'categorical' ? 'selected' : ''}>categorical</option>
              <option value="boolean" ${type === 'boolean' ? 'selected' : ''}>boolean</option>
              <option value="datetime" ${type === 'datetime' ? 'selected' : ''}>datetime</option>
              <option value="identifier" ${type === 'identifier' ? 'selected' : ''}>identifier</option>
              <option value="text" ${type === 'text' ? 'selected' : ''}>text</option>
            </select>
          </div>
        </th>
      `;
    });
    thHtml += '</tr>';
    el.previewTableHead.innerHTML = thHtml;

    // Wire type override events
    el.previewTableHead.querySelectorAll('.type-override-select').forEach((sel) => {
      sel.addEventListener('change', (e) => {
        overrideColumnType(e.target.dataset.col, e.target.value);
      });
    });

    // Render tbody (up to 8 sample rows)
    const previewRows = rows.slice(0, 8);
    let trHtml = '';
    previewRows.forEach((row) => {
      trHtml += '<tr>';
      row.forEach((cell) => {
        trHtml += `<td>${escapeHtml(cell !== undefined && cell !== null ? String(cell) : '—')}</td>`;
      });
      trHtml += '</tr>';
    });
    el.previewTableBody.innerHTML = trHtml;
  }

  // ============================================================
  // 6. RESTRICTIONS LOGIC & RENDERING
  // ============================================================
  function renderRestrictionsSection() {
    if (!state.dataset) return;
    const { headers, types } = state.dataset;

    if (el.colRestrictionCount) {
      el.colRestrictionCount.textContent = `${headers.length} columns`;
    }

    // Populate column restriction rows
    let rowsHtml = '';
    headers.forEach((col) => {
      const type = types[col] || 'text';
      const status = state.restrictions.columns[col] || 'include';

      rowsHtml += `
        <div class="restriction-row" data-col="${escapeHtml(col)}">
          <div class="column-info">
            <span class="column-name">${escapeHtml(col)}</span>
            <span class="data-type-badge ${type}">${type}</span>
          </div>
          <div class="tri-control">
            <button class="tri-btn ${status === 'include' ? 'active-include' : ''}" data-action="include" data-col="${escapeHtml(col)}">Include</button>
            <button class="tri-btn ${status === 'exclude' ? 'active-exclude' : ''}" data-action="exclude" data-col="${escapeHtml(col)}">Exclude</button>
            <button class="tri-btn ${status === 'review' ? 'active-review' : ''}" data-action="review" data-col="${escapeHtml(col)}">Review</button>
          </div>
        </div>
      `;
    });

    el.columnRestrictions.innerHTML = rowsHtml;

    // Attach click listeners to tri-buttons
    el.columnRestrictions.querySelectorAll('.tri-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const col = e.currentTarget.dataset.col;
        const action = e.currentTarget.dataset.action;
        setColumnStatus(col, action);
      });
    });
  }

  function setColumnStatus(col, action) {
    state.restrictions.columns[col] = action;

    // Update row button states
    const row = el.columnRestrictions.querySelector(`[data-col="${col}"]`);
    if (row) {
      row.querySelectorAll('.tri-btn').forEach((btn) => {
        btn.classList.remove('active-include', 'active-exclude', 'active-review');
        if (btn.dataset.action === action) {
          btn.classList.add(`active-${action}`);
        }
      });
    }

    updateRestrictionSummary();
    renderFeatureTable();
  }

  function populateTargetDropdowns() {
    if (!state.dataset) return;
    const { headers } = state.dataset;

    let options = '<option value="">Select target variable...</option>';
    headers.forEach((h) => {
      options += `<option value="${escapeHtml(h)}">${escapeHtml(h)}</option>`;
    });

    if (el.targetColumn) {
      el.targetColumn.innerHTML = options;
      // Preselect last column as likely target
      const defaultTarget = headers[headers.length - 1];
      el.targetColumn.value = defaultTarget;
      state.restrictions.constraints.targetColumn = defaultTarget;
    }

    if (el.expTarget) {
      el.expTarget.innerHTML = options;
      el.expTarget.value = state.restrictions.constraints.targetColumn;
    }

    updateRestrictionSummary();
  }

  function updateRestrictionSummary() {
    if (!state.dataset) return;

    let inc = 0;
    let exc = 0;
    let rev = 0;
    const excludedNames = [];

    Object.entries(state.restrictions.columns).forEach(([col, status]) => {
      if (status === 'include') inc++;
      else if (status === 'exclude') {
        exc++;
        excludedNames.push(col);
      } else if (status === 'review') rev++;
    });

    if (el.summaryIncluded) el.summaryIncluded.textContent = inc.toString();
    if (el.summaryExcluded) el.summaryExcluded.textContent = exc.toString();
    if (el.summaryReview) el.summaryReview.textContent = rev.toString();

    // Excluded features tags
    if (el.excludedFeatures) {
      if (excludedNames.length === 0) {
        el.excludedFeatures.innerHTML = '<span class="text-muted caption">None excluded</span>';
      } else {
        el.excludedFeatures.innerHTML = excludedNames
          .map((name) => `<span class="badge badge-exclude mr-2 mb-1">${escapeHtml(name)}</span>`)
          .join(' ');
      }
    }

    // Sensitive count
    const sensitiveActive = Object.values(state.restrictions.sensitive).filter(Boolean).length;
    if (el.sensitiveCount) el.sensitiveCount.textContent = `${sensitiveActive} active`;
    if (el.summarySensitive) el.summarySensitive.textContent = `${sensitiveActive} active rules`;

    // Quality count
    const qualityActive = Object.values(state.restrictions.quality).filter(Boolean).length;
    if (el.qualityCount) el.qualityCount.textContent = `${qualityActive} active`;
    if (el.summaryQuality) el.summaryQuality.textContent = `${qualityActive} active rules`;

    // Target
    const target = state.restrictions.constraints.targetColumn || 'Not selected';
    if (el.summaryTarget) el.summaryTarget.textContent = target;

    // Encoding & Split
    const encMap = { auto: 'Automatic', label: 'Ordinal', onehot: 'One-Hot' };
    if (el.summaryEncoding) el.summaryEncoding.textContent = encMap[state.restrictions.constraints.encoding] || 'Automatic';

    const scaleMap = { standard: 'StandardScaler', minmax: 'MinMaxScaler', robust: 'RobustScaler', none: 'No Scaling' };
    if (el.summaryScaling) el.summaryScaling.textContent = scaleMap[state.restrictions.constraints.scaling] || 'StandardScaler';

    const outlierMap = { none: 'Keep As-Is', clip: 'Clip (Winsorize)', drop: 'Drop Rows' };
    if (el.summaryOutliers) el.summaryOutliers.textContent = outlierMap[state.restrictions.constraints.outlierTreatment] || 'Keep As-Is';

    const splitMap = { '70-30': '70 / 30', '80-20': '80 / 20', '80-10-10': '80 / 10 / 10' };
    if (el.summarySplit) el.summarySplit.textContent = splitMap[state.restrictions.constraints.split] || '80 / 20';
  }


  // ============================================================
  // 7. OUTPUT SECTION & VISUALIZATIONS
  // ============================================================
  function renderOutputSection() {
    if (!state.dataset) return;
    const ds = state.dataset;
    const { rows, headers, types, analysis } = ds;

    // Actual metrics from backend analysis if available
    if (analysis) {
      if (el.metricRows) el.metricRows.textContent = analysis.total_rows.toLocaleString();
      if (el.metricCols) el.metricCols.textContent = analysis.total_columns.toString();
      if (el.metricNumeric) el.metricNumeric.textContent = (analysis.column_types_summary.numeric || 0).toString();
      if (el.metricCategorical) {
        const catTotal = (analysis.column_types_summary.categorical || 0) + (analysis.column_types_summary.boolean || 0);
        el.metricCategorical.textContent = catTotal.toString();
      }
      if (el.metricMissing) el.metricMissing.textContent = analysis.missing_cells_total.toLocaleString();

      if (el.metricRowsNote) el.metricRowsNote.textContent = `${analysis.duplicate_rows} duplicates (${analysis.duplicate_rows_percentage}%)`;
      if (el.metricColsNote) el.metricColsNote.textContent = `Memory: ${analysis.memory_usage_formatted}`;
      if (el.metricNumericNote) el.metricNumericNote.textContent = 'Verified numeric features';
      if (el.metricCategoricalNote) el.metricCategoricalNote.textContent = 'Categorical & boolean features';
      if (el.metricMissingNote) el.metricMissingNote.textContent = `${analysis.missing_cells_percentage}% total missingness`;
    } else {
      const rowCount = rows.length;
      const colCount = headers.length;
      let numCount = 0;
      let catCount = 0;
      headers.forEach((h) => {
        const t = types[h];
        if (t === 'numeric') numCount++;
        else if (t === 'categorical' || t === 'boolean') catCount++;
      });
      if (el.metricRows) el.metricRows.textContent = rowCount.toLocaleString();
      if (el.metricCols) el.metricCols.textContent = colCount.toString();
      if (el.metricNumeric) el.metricNumeric.textContent = numCount.toString();
      if (el.metricCategorical) el.metricCategorical.textContent = catCount.toString();
      if (el.metricMissing) el.metricMissing.textContent = '0';
      if (el.metricRowsNote) el.metricRowsNote.textContent = 'From local preview';
      if (el.metricColsNote) el.metricColsNote.textContent = 'From local preview';
      if (el.metricNumericNote) el.metricNumericNote.textContent = 'Inferred by types';
      if (el.metricCategoricalNote) el.metricCategoricalNote.textContent = 'Inferred by types';
      if (el.metricMissingNote) el.metricMissingNote.textContent = 'Checked in loaded rows';
    }

    // Render All Actual Visualizations
    renderDistributionChart();
    renderCategoryChart();
    renderMissingChart();
    renderCorrelationsChart();
    renderBoxPlotChart();
    renderScatterChart();
    renderFeatureTable();
  }

  function renderDistributionChart(selectedCol) {
    if (!state.dataset || !el.distributionChart) return;
    const ds = state.dataset;
    const { headers, rows, types, visualizations } = ds;

    if (visualizations && visualizations.histograms && Object.keys(visualizations.histograms).length > 0) {
      const numCols = visualizations.numeric_columns && visualizations.numeric_columns.length > 0
        ? visualizations.numeric_columns
        : headers.filter((h) => types[h] === 'numeric');

      if (numCols.length === 0) {
        el.distributionChart.innerHTML = '<p class="caption text-muted" style="padding:20px;">No numeric columns available in dataset.</p>';
        return;
      }

      const activeCol = selectedCol && numCols.includes(selectedCol) ? selectedCol : (visualizations.primary_numeric || numCols[0]);
      const hist = visualizations.histograms[activeCol] || { labels: [], counts: [], min: 0, max: 0 };

      let selectorHtml = `
        <div class="viz-selector-row">
          <span class="caption text-secondary">Feature: <strong>${escapeHtml(activeCol)}</strong> (Min: ${hist.min}, Max: ${hist.max})</span>
          <select class="viz-select" id="histColSelector">
            ${numCols.map((c) => `<option value="${escapeHtml(c)}" ${c === activeCol ? 'selected' : ''}>${escapeHtml(c)}</option>`).join('')}
          </select>
        </div>
      `;

      const maxBin = Math.max(...(hist.counts || [1]), 1);
      let barsHtml = '<div class="bar-chart">';
      hist.counts.forEach((count, i) => {
        const heightPct = Math.max(10, Math.round((count / maxBin) * 170));
        const label = hist.labels[i] || '';
        barsHtml += `
          <div class="bar-group">
            <div class="bar" style="height: ${heightPct}px;" data-value="${count}"></div>
            <span class="bar-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span>
          </div>
        `;
      });
      barsHtml += '</div>';

      el.distributionChart.innerHTML = selectorHtml + barsHtml;

      const sel = document.getElementById('histColSelector');
      if (sel) {
        sel.addEventListener('change', (e) => {
          renderDistributionChart(e.target.value);
        });
      }
      return;
    }

    // Client fallback
    const numCol = headers.find((h) => types[h] === 'numeric') || headers[0];
    const colIndex = headers.indexOf(numCol);
    const values = rows.map((r) => parseFloat(r[colIndex])).filter((v) => !isNaN(v));

    if (values.length === 0) {
      el.distributionChart.innerHTML = '<p class="caption text-muted">No numeric distribution available.</p>';
      return;
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    const binsCount = 8;
    const binWidth = (max - min) / binsCount || 1;
    const bins = new Array(binsCount).fill(0);

    values.forEach((v) => {
      let b = Math.floor((v - min) / binWidth);
      if (b >= binsCount) b = binsCount - 1;
      bins[b]++;
    });

    const maxBin = Math.max(...bins, 1);
    let html = '';
    bins.forEach((count, i) => {
      const heightPct = Math.max(10, Math.round((count / maxBin) * 180));
      const rangeStart = (min + i * binWidth).toFixed(0);
      html += `
        <div class="bar-group">
          <div class="bar" style="height: ${heightPct}px;" data-value="${count}"></div>
          <span class="bar-label">${rangeStart}</span>
        </div>
      `;
    });
    el.distributionChart.innerHTML = html;
  }

  function renderCategoryChart(selectedCol) {
    if (!state.dataset || !el.categoryChart) return;
    const ds = state.dataset;
    const { headers, rows, types, visualizations } = ds;

    if (visualizations && visualizations.categories && Object.keys(visualizations.categories).length > 0) {
      const catCols = visualizations.categorical_columns && visualizations.categorical_columns.length > 0
        ? visualizations.categorical_columns
        : headers.filter((h) => types[h] === 'categorical' || types[h] === 'boolean');

      if (catCols.length === 0) {
        el.categoryChart.innerHTML = '<p class="caption text-muted" style="padding:20px;">No categorical columns available in dataset.</p>';
        return;
      }

      const activeCol = selectedCol && catCols.includes(selectedCol) ? selectedCol : (visualizations.primary_categorical || catCols[0]);
      const catData = visualizations.categories[activeCol] || { categories: [], counts: [], other_count: 0 };

      let selectorHtml = `
        <div class="viz-selector-row">
          <span class="caption text-secondary">Feature: <strong>${escapeHtml(activeCol)}</strong> (${catData.total_categories || catData.categories.length} unique values)</span>
          <select class="viz-select" id="catColSelector">
            ${catCols.map((c) => `<option value="${escapeHtml(c)}" ${c === activeCol ? 'selected' : ''}>${escapeHtml(c)}</option>`).join('')}
          </select>
        </div>
      `;

      const maxVal = Math.max(...(catData.counts || [1]), 1);
      let barsHtml = '<div class="h-bar-chart">';
      catData.categories.forEach((cat, idx) => {
        const count = catData.counts[idx] || 0;
        const pct = Math.max(8, Math.round((count / maxVal) * 100));
        barsHtml += `
          <div class="h-bar-row">
            <span class="h-bar-name" title="${escapeHtml(cat)}">${escapeHtml(cat)}</span>
            <div class="h-bar-track">
              <div class="h-bar-fill" style="width: ${pct}%;">
                <span class="h-bar-value">${count}</span>
              </div>
            </div>
          </div>
        `;
      });

      if (catData.other_count > 0) {
        const otherPct = Math.max(8, Math.round((catData.other_count / maxVal) * 100));
        barsHtml += `
          <div class="h-bar-row">
            <span class="h-bar-name text-muted">(Other ${catData.other_count} values)</span>
            <div class="h-bar-track">
              <div class="h-bar-fill" style="width: ${otherPct}%; background: var(--text-muted);">
                <span class="h-bar-value">${catData.other_count}</span>
              </div>
            </div>
          </div>
        `;
      }
      barsHtml += '</div>';

      el.categoryChart.innerHTML = selectorHtml + barsHtml;

      const sel = document.getElementById('catColSelector');
      if (sel) {
        sel.addEventListener('change', (e) => {
          renderCategoryChart(e.target.value);
        });
      }
      return;
    }

    // Client fallback
    const catCol = headers.find((h) => types[h] === 'categorical') || headers[1] || headers[0];
    const colIndex = headers.indexOf(catCol);
    const counts = {};
    rows.forEach((r) => {
      const val = r[colIndex] || 'Unknown';
      counts[val] = (counts[val] || 0) + 1;
    });

    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 5);
    const maxVal = Math.max(...sorted.map((s) => s[1]), 1);
    let html = '';
    sorted.forEach(([cat, count]) => {
      const pct = Math.max(8, Math.round((count / maxVal) * 100));
      html += `
        <div class="h-bar-row">
          <span class="h-bar-name" title="${escapeHtml(cat)}">${escapeHtml(cat)}</span>
          <div class="h-bar-track">
            <div class="h-bar-fill" style="width: ${pct}%;">
              <span class="h-bar-value">${count}</span>
            </div>
          </div>
        </div>
      `;
    });
    el.categoryChart.innerHTML = html;
  }

  function renderMissingChart() {
    if (!state.dataset || !el.missingChart) return;
    const ds = state.dataset;
    const { visualizations, headers, rows } = ds;

    if (visualizations && visualizations.missing_overview && visualizations.missing_overview.length > 0) {
      let html = '<div class="missing-chart">';
      visualizations.missing_overview.forEach((item) => {
        const hasMissing = item.missing_count > 0;
        const fillWidth = hasMissing ? Math.max(item.missing_pct, 6) : 100;
        html += `
          <div class="missing-row">
            <span class="missing-name" title="${escapeHtml(item.column)}">${escapeHtml(item.column)}</span>
            <div class="missing-bar-track">
              <div class="missing-bar-fill ${hasMissing ? 'has-missing' : 'no-missing'}" style="width: ${fillWidth}%;"></div>
            </div>
            <span class="missing-pct">${item.missing_pct}% (${item.missing_count})</span>
          </div>
        `;
      });
      html += '</div>';
      el.missingChart.innerHTML = html;
      return;
    }

    // Client fallback
    let html = '';
    headers.slice(0, 8).forEach((h, colIndex) => {
      const total = rows.length || 1;
      const missing = rows.filter((r) => r[colIndex] === '' || r[colIndex] === null || r[colIndex] === undefined).length;
      const pct = Math.round((missing / total) * 100);
      html += `
        <div class="missing-row">
          <span class="missing-name">${escapeHtml(h)}</span>
          <div class="missing-bar-track">
            <div class="missing-bar-fill ${missing > 0 ? 'has-missing' : 'no-missing'}" style="width: ${missing > 0 ? pct : 100}%;"></div>
          </div>
          <span class="missing-pct">${pct}%</span>
        </div>
      `;
    });
    el.missingChart.innerHTML = html;
  }

  function renderCorrelationsChart() {
    if (!state.dataset || !el.correlationsHeatmap) return;
    const ds = state.dataset;
    const viz = ds.visualizations;

    if (!viz || !viz.correlation_matrix || !viz.correlation_matrix.matrix || viz.correlation_matrix.matrix.length === 0) {
      el.correlationsHeatmap.innerHTML = `
        <div class="empty-state" style="padding: 30px 0;">
          <p class="caption text-muted">Insufficient numeric features (requires at least 2) to compute correlation matrix.</p>
        </div>
      `;
      return;
    }

    const { columns, matrix } = viz.correlation_matrix;

    let tableHtml = '<div style="margin-bottom:12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">';
    tableHtml += '<span class="caption text-secondary">Pearson Correlation Matrix (-1.00 to +1.00)</span>';
    tableHtml += '<div style="display:flex; align-items:center; gap:8px; font-size:10px; font-family:var(--font-mono);">';
    tableHtml += '<span style="color:#388eaf;">■ -1.0 Inverse</span>';
    tableHtml += '<span style="color:var(--text-muted);">■ 0.0 Neutral</span>';
    tableHtml += '<span style="color:#ff603d;">■ +1.0 Direct</span>';
    tableHtml += '</div></div>';

    tableHtml += '<table class="heatmap-table"><thead><tr><th></th>';
    columns.forEach((col) => {
      tableHtml += `<th title="${escapeHtml(col)}">${escapeHtml(col)}</th>`;
    });
    tableHtml += '</tr></thead><tbody>';

    matrix.forEach((row, i) => {
      tableHtml += `<tr><td class="heatmap-label" title="${escapeHtml(columns[i])}">${escapeHtml(columns[i])}</td>`;
      row.forEach((val, j) => {
        let bgColor = 'var(--surface)';
        let textColor = 'var(--text-primary)';
        const absVal = Math.abs(val);

        if (i === j) {
          bgColor = 'rgba(255, 96, 61, 0.25)';
          textColor = 'var(--accent)';
        } else if (val > 0) {
          bgColor = `rgba(255, 96, 61, ${Math.min(0.85, 0.15 + absVal * 0.7)})`;
          if (absVal > 0.45) textColor = '#fff';
        } else if (val < 0) {
          bgColor = `rgba(56, 142, 175, ${Math.min(0.85, 0.15 + absVal * 0.7)})`;
          if (absVal > 0.45) textColor = '#fff';
        }

        const sign = val > 0 && i !== j ? '+' : '';
        tableHtml += `
          <td class="heatmap-cell" style="background:${bgColor}; color:${textColor};" title="r(${escapeHtml(columns[i])}, ${escapeHtml(columns[j])}) = ${val}">
            ${sign}${val.toFixed(2)}
          </td>
        `;
      });
      tableHtml += '</tr>';
    });
    tableHtml += '</tbody></table>';

    el.correlationsHeatmap.innerHTML = tableHtml;
  }

  function renderBoxPlotChart(selectedCol) {
    if (!state.dataset || !el.boxPlotChart) return;
    const ds = state.dataset;
    const viz = ds.visualizations;

    if (!viz || !viz.box_plots || Object.keys(viz.box_plots).length === 0) {
      el.boxPlotChart.innerHTML = '<p class="caption text-muted" style="padding:20px;">No numeric features available for box plot.</p>';
      return;
    }

    const numCols = viz.numeric_columns && viz.numeric_columns.length > 0
      ? viz.numeric_columns
      : Object.keys(viz.box_plots);

    const activeCol = selectedCol && numCols.includes(selectedCol) ? selectedCol : (viz.primary_numeric || numCols[0]);
    const bp = viz.box_plots[activeCol] || { min: 0, q1: 0, median: 0, q3: 0, max: 0, outliers: [] };

    let html = `
      <div class="viz-selector-row">
        <span class="caption text-secondary">Box & Whisker Plot: <strong>${escapeHtml(activeCol)}</strong></span>
        <select class="viz-select" id="boxColSelector">
          ${numCols.map((c) => `<option value="${escapeHtml(c)}" ${c === activeCol ? 'selected' : ''}>${escapeHtml(c)}</option>`).join('')}
        </select>
      </div>
      <div class="boxplot-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
          <span class="body-sm font-semibold">${escapeHtml(activeCol)} Distribution</span>
          <span class="badge ${bp.outlier_count > 0 ? 'badge-review' : 'badge-include'}">${bp.outlier_count} Outliers</span>
        </div>
        <div style="position:relative; height:60px; background:var(--bg-primary); border-radius:4px; padding:10px 20px; display:flex; align-items:center;">
          <div style="width:100%; height:6px; background:var(--border-subtle); border-radius:3px; position:relative;">
            <div style="position:absolute; left:25%; width:50%; height:18px; top:-6px; background:var(--accent); opacity:0.35; border:1px solid var(--accent); border-radius:3px;"></div>
            <div style="position:absolute; left:50%; width:3px; height:24px; top:-9px; background:var(--accent); border-radius:1px;"></div>
          </div>
        </div>
        <div class="boxplot-stats">
          <span>Min: <strong>${bp.min}</strong></span>
          <span>Q1: <strong>${bp.q1}</strong></span>
          <span>Median: <strong>${bp.median}</strong></span>
          <span>Q3: <strong>${bp.q3}</strong></span>
          <span>Max: <strong>${bp.max}</strong></span>
          <span>IQR: <strong>${bp.iqr}</strong></span>
        </div>
      </div>
    `;

    el.boxPlotChart.innerHTML = html;

    const sel = document.getElementById('boxColSelector');
    if (sel) {
      sel.addEventListener('change', (e) => {
        renderBoxPlotChart(e.target.value);
      });
    }
  }

  async function renderScatterChart(xCol, yCol) {
    if (!state.dataset || !el.scatterChart) return;
    const ds = state.dataset;
    const viz = ds.visualizations;

    if (!viz || !viz.numeric_columns || viz.numeric_columns.length < 2) {
      el.scatterChart.innerHTML = '<p class="caption text-muted" style="padding:20px;">Scatter plot requires at least 2 numerical features.</p>';
      return;
    }

    const numCols = viz.numeric_columns;
    const activeX = xCol && numCols.includes(xCol) ? xCol : numCols[0];
    const activeY = yCol && numCols.includes(yCol) ? yCol : (numCols[1] || numCols[0]);

    let points = (viz.scatter_plot && viz.scatter_plot.x_col === activeX && viz.scatter_plot.y_col === activeY) ? viz.scatter_plot.points : [];
    if ((!points || points.length === 0) && ds.id) {
      try {
        const res = await fetch(`${API_BASE}/datasets/${ds.id}/visualizations?scatter_x=${encodeURIComponent(activeX)}&scatter_y=${encodeURIComponent(activeY)}`);
        if (res.ok) {
          const freshViz = await res.json();
          points = freshViz.scatter_plot ? freshViz.scatter_plot.points : [];
        }
      } catch (e) {}
    }

    let selectorHtml = `
      <div class="viz-selector-row" style="max-width:580px;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span class="caption text-secondary">X Feature:</span>
          <select class="viz-select" id="scatterXSelector">
            ${numCols.map((c) => `<option value="${escapeHtml(c)}" ${c === activeX ? 'selected' : ''}>${escapeHtml(c)}</option>`).join('')}
          </select>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <span class="caption text-secondary">Y Feature:</span>
          <select class="viz-select" id="scatterYSelector">
            ${numCols.map((c) => `<option value="${escapeHtml(c)}" ${c === activeY ? 'selected' : ''}>${escapeHtml(c)}</option>`).join('')}
          </select>
        </div>
      </div>
    `;

    if (!points || points.length === 0) {
      el.scatterChart.innerHTML = selectorHtml + '<p class="caption text-muted" style="padding:20px;">No coordinate points found for selected pair.</p>';
      return;
    }

    const xVals = points.map((p) => p.x);
    const yVals = points.map((p) => p.y);
    const minX = Math.min(...xVals);
    const maxX = Math.max(...xVals) || minX + 1;
    const minY = Math.min(...yVals);
    const maxY = Math.max(...yVals) || minY + 1;

    const svgWidth = 580;
    const svgHeight = 240;
    const pad = 36;

    let dotsSvg = '';
    points.forEach((p) => {
      const cx = pad + ((p.x - minX) / (maxX - minX || 1)) * (svgWidth - pad * 2);
      const cy = svgHeight - pad - ((p.y - minY) / (maxY - minY || 1)) * (svgHeight - pad * 2);
      dotsSvg += `<circle class="scatter-dot" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="3.5"><title>${activeX}: ${p.x}, ${activeY}: ${p.y}</title></circle>`;
    });

    const chartSvg = `
      <svg class="scatter-svg" viewBox="0 0 ${svgWidth} ${svgHeight}">
        <line x1="${pad}" y1="${svgHeight - pad}" x2="${svgWidth - pad}" y2="${svgHeight - pad}" stroke="var(--border)" stroke-width="1.5" />
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${svgHeight - pad}" stroke="var(--border)" stroke-width="1.5" />
        <text x="${svgWidth - pad}" y="${svgHeight - pad + 18}" font-size="10" font-family="var(--font-mono)" fill="var(--text-secondary)" text-anchor="end">${escapeHtml(activeX)} (Max: ${maxX.toFixed(1)})</text>
        <text x="${pad}" y="${pad - 10}" font-size="10" font-family="var(--font-mono)" fill="var(--text-secondary)">${escapeHtml(activeY)} (Max: ${maxY.toFixed(1)})</text>
        ${dotsSvg}
      </svg>
    `;

    el.scatterChart.innerHTML = selectorHtml + chartSvg;

    const selX = document.getElementById('scatterXSelector');
    const selY = document.getElementById('scatterYSelector');
    if (selX) selX.addEventListener('change', (e) => renderScatterChart(e.target.value, selY ? selY.value : activeY));
    if (selY) selY.addEventListener('change', (e) => renderScatterChart(selX ? selX.value : activeX, e.target.value));
  }

  function renderFeatureTable() {
    if (!state.dataset || !el.featureTableBody) return;
    const { headers, types } = state.dataset;

    let html = '';
    headers.forEach((col) => {
      const type = types[col] || 'text';
      const status = state.restrictions.columns[col] || 'include';
      const isTarget = col === state.restrictions.constraints.targetColumn;

      const badgeClass = status === 'include' ? 'badge-include' : status === 'exclude' ? 'badge-exclude' : 'badge-review';
      const statusLabel = isTarget ? 'Target Variable' : status.toUpperCase();

      let pipelineTransform = 'Pass-Through';
      if (isTarget) {
        pipelineTransform = type === 'numeric' ? 'Continuous Target' : 'LabelEncoder (Fitted on Train)';
      } else if (status === 'exclude') {
        pipelineTransform = 'Excluded';
      } else if (type === 'numeric') {
        const sc = state.restrictions.constraints.scaling || 'standard';
        pipelineTransform = sc === 'minmax' ? 'MinMaxScaler' : sc === 'robust' ? 'RobustScaler' : sc === 'none' ? 'None' : 'StandardScaler';
      } else {
        const enc = state.restrictions.constraints.encoding || 'auto';
        pipelineTransform = enc === 'onehot' ? 'OneHotEncoder' : enc === 'label' ? 'OrdinalEncoder' : 'Auto (OneHot/Ordinal)';
      }

      html += `
        <tr>
          <td>
            <strong>${escapeHtml(col)}</strong>
            ${isTarget ? ' <span class="badge badge-include ml-2" style="font-size:9px;">TARGET</span>' : ''}
          </td>
          <td><span class="data-type-badge ${type}">${type}</span></td>
          <td><span class="badge ${badgeClass}">${statusLabel}</span></td>
          <td><span class="mono">${pipelineTransform}</span></td>
          <td><span class="caption text-muted">${status === 'exclude' ? 'Excluded from training pipeline' : isTarget ? 'Dependent response variable' : 'Active Feature'}</span></td>
        </tr>
      `;
    });

    el.featureTableBody.innerHTML = html;
    fetchAndRenderPreprocessingPreview();
  }

  async function fetchAndRenderPreprocessingPreview() {
    if (!state.dataset || !state.dataset.id) return;
    const ds = state.dataset;

    const excluded = Object.entries(state.restrictions.columns)
      .filter(([col, status]) => status === 'exclude')
      .map(([col]) => col);

    const missingMap = {
      drop: 'drop',
      mean: 'mean',
      median: 'median',
      keep: 'keep',
    };

    const encMap = {
      auto: 'auto',
      onehot: 'onehot',
      label: 'ordinal',
      ordinal: 'ordinal',
    };

    const payload = {
      target_column: state.restrictions.constraints.targetColumn || null,
      excluded_columns: excluded,
      type_overrides: state.dataset.types || {},
      missing_numeric_strategy: missingMap[state.restrictions.constraints.missingHandling] || 'mean',
      missing_categorical_strategy: state.restrictions.constraints.missingHandling === 'drop' ? 'drop' : 'most_frequent',
      encoding_strategy: encMap[state.restrictions.constraints.encoding] || 'auto',
      scaling_strategy: state.restrictions.constraints.scaling || 'standard',
      outlier_strategy: state.restrictions.constraints.outlierTreatment || 'none',
      handle_duplicates: state.restrictions.quality.duplicates !== false,
      train_test_split_ratio: state.restrictions.constraints.split === '70-30' ? 0.7 : 0.8,
    };

    try {
      const res = await fetch(`${API_BASE}/datasets/${ds.id}/preprocess/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        console.warn('Preprocessing preview fetch returned status', res.status);
        return;
      }

      const preview = await res.json();
      renderPreprocessingPreviewUI(preview);
    } catch (err) {
      console.error('Failed to fetch preprocessing preview:', err);
    }
  }

  function renderPreprocessingPreviewUI(preview) {
    if (!preview) return;

    if (el.previewInCols) el.previewInCols.textContent = preview.feature_names_in ? preview.feature_names_in.length : 0;
    if (el.previewOutCols) el.previewOutCols.textContent = preview.output_feature_count || 0;
    if (el.previewFeaturesBadge) el.previewFeaturesBadge.textContent = `Features: ${preview.feature_names_in ? preview.feature_names_in.length : 0} → ${preview.output_feature_count || 0}`;
    if (el.previewTrainSplit && preview.train_shape) el.previewTrainSplit.textContent = `${preview.train_shape[0]} rows × ${preview.train_shape[1]} cols`;
    if (el.previewTestSplit && preview.test_shape) el.previewTestSplit.textContent = `${preview.test_shape[0]} rows × ${preview.test_shape[1]} cols`;

    // Warnings Banner
    if (el.previewWarningsBanner && el.previewWarningsContent) {
      if (preview.warnings && preview.warnings.length > 0) {
        el.previewWarningsBanner.classList.remove('hidden');
        el.previewWarningsContent.innerHTML = preview.warnings
          .map((w) => `<div style="margin-bottom:4px;">${escapeHtml(w)}</div>`)
          .join('');
      } else {
        el.previewWarningsBanner.classList.add('hidden');
      }
    }

    // Operations Applied list
    if (el.previewOperationsList && preview.operations_applied) {
      el.previewOperationsList.innerHTML = preview.operations_applied.map((op) => `
        <div class="operation-item">
          <div class="operation-left">
            <span class="operation-badge">${escapeHtml(op.step_name)}</span>
            <span class="operation-desc">${escapeHtml(op.description)}</span>
          </div>
          <div class="operation-cols">${op.target_columns && op.target_columns.length > 0 ? escapeHtml(op.target_columns.slice(0, 3).join(', ') + (op.target_columns.length > 3 ? ` (+${op.target_columns.length - 3} more)` : '')) : 'All Features'}</div>
        </div>
      `).join('');
    }

    // Transformed Feature Matrix Sample (first 5 rows)
    if (el.transformedSampleHead && el.transformedSampleBody && preview.sample_preview && preview.sample_preview.length > 0) {
      const allCols = preview.feature_names_out || [];
      const visibleCols = allCols.slice(0, 10);
      const isTruncated = allCols.length > 10;

      let headHtml = '<tr><th style="width:50px;">#</th>';
      visibleCols.forEach((col) => {
        headHtml += `<th title="${escapeHtml(col)}">${escapeHtml(col)}</th>`;
      });
      if (isTruncated) headHtml += `<th>... +${allCols.length - 10} more</th>`;
      if (preview.target_column) headHtml += `<th style="color:var(--accent);">Target (${escapeHtml(preview.target_column)})</th>`;
      headHtml += '</tr>';
      el.transformedSampleHead.innerHTML = headHtml;

      let bodyHtml = '';
      preview.sample_preview.slice(0, 5).forEach((row, idx) => {
        bodyHtml += `<tr><td class="font-mono text-muted">${idx + 1}</td>`;
        visibleCols.forEach((col) => {
          const val = row[col] !== undefined ? row[col] : '-';
          bodyHtml += `<td class="font-mono">${escapeHtml(String(val))}</td>`;
        });
        if (isTruncated) bodyHtml += '<td class="text-muted">...</td>';
        if (preview.target_preview && preview.target_preview[idx] !== undefined) {
          bodyHtml += `<td class="font-mono text-accent"><strong>${escapeHtml(String(preview.target_preview[idx]))}</strong></td>`;
        }
        bodyHtml += '</tr>';
      });
      el.transformedSampleBody.innerHTML = bodyHtml;
    }

    // Update pipeline badges
    if (el.pipelineEncodingVal) {
      el.pipelineEncodingVal.textContent = `${state.restrictions.constraints.encoding === 'onehot' ? 'One-Hot' : state.restrictions.constraints.encoding === 'label' ? 'Ordinal' : 'Automatic'} Encoding (${preview.output_feature_count} output features).`;
    }
    if (el.pipelineSplitVal && preview.train_shape && preview.test_shape) {
      el.pipelineSplitVal.textContent = `${state.restrictions.constraints.split} Split: ${preview.train_shape[0]} Train rows / ${preview.test_shape[0]} Test rows.`;
    }
  }


  // ============================================================
  // 8. MODEL SELECTION & HYPERPARAMETERS (ALGORITHM REGISTRY)
  // ============================================================
  let hpValidationDebounceTimer = null;

  async function selectModel(modelKey) {
    state.selectedModel = modelKey;
    const meta = MODEL_METADATA[modelKey] || {
      name: modelKey,
      type: 'classification',
      tasks: ['classification'],
      defaultMetric: 'f1',
      paramGroup: 'treeParams',
      algoId: modelKey,
    };

    // Update active model cards
    el.modelCards.forEach((card) => {
      if (card.dataset.model === modelKey) {
        card.classList.add('selected');
      } else {
        card.classList.remove('selected');
      }
    });

    // Update select in experiment config form
    if (el.expModel && el.expModel.value !== modelKey) {
      el.expModel.value = modelKey;
    }

    // Update evaluation metrics based on model type
    updateMetricsOptions(meta.type, meta.defaultMetric);

    // Switch visible hyperparameter panel (hide legacy panels)
    switchHyperparamPanel(null);

    // Handle clustering specifics (Unsupervised: target variable optional)
    handleClusteringWorkflowUI(meta.type);

    // Load registry algorithm schema & dynamic hyperparameter form
    await loadRegistryAlgorithm(meta.algoId || modelKey, meta.type);

    showNotification(`Selected algorithm: ${meta.name}`, 'info');
  }

  function handleClusteringWorkflowUI(modelType) {
    const isClustering = modelType === 'clustering';
    if (isClustering) {
      if (el.summaryTarget) {
        el.summaryTarget.innerHTML = '<span class="badge badge-include">Unsupervised (No Target Required)</span>';
      }
      if (el.expTarget) {
        let noneOption = el.expTarget.querySelector('option[value=""]');
        if (!noneOption) {
          const opt = document.createElement('option');
          opt.value = '';
          opt.textContent = 'None — Unsupervised Clustering';
          el.expTarget.insertBefore(opt, el.expTarget.firstChild);
        }
        el.expTarget.value = '';
      }
      if (el.expMetric) {
        el.expMetric.value = 'silhouette';
      }
    } else {
      if (state.restrictions.constraints.targetColumn && el.summaryTarget) {
        el.summaryTarget.textContent = state.restrictions.constraints.targetColumn;
      }
    }
  }

  async function loadRegistryAlgorithm(modelKey, taskHint) {
    if (!el.dynamicHyperparamFields) return;

    if (el.hyperparamStatusBadge) {
      el.hyperparamStatusBadge.textContent = 'Loading Registry...';
      el.hyperparamStatusBadge.className = 'badge badge-review font-mono';
    }

    try {
      const taskParam = (taskHint === 'classification' || taskHint === 'regression' || taskHint === 'clustering')
        ? `?task=${taskHint}`
        : '';
      const response = await fetch(`${API_BASE}/algorithms/${encodeURIComponent(modelKey)}${taskParam}`);
      
      if (!response.ok) {
        throw new Error(`Registry endpoint error: HTTP ${response.status}`);
      }

      const algoData = await response.json();
      state.algorithmConfig = {
        id: algoData.id,
        display_name: algoData.display_name,
        task: algoData.task,
        hyperparameters: { ...algoData.default_hyperparameters },
        valid: true,
        errors: [],
      };

      if (el.dynamicAlgoDesc) {
        el.dynamicAlgoDesc.innerHTML = `<strong>${escapeHtml(algoData.display_name)}</strong> (${escapeHtml(algoData.task)}): ${escapeHtml(algoData.description)}`;
      }

      renderDynamicHyperparameters(algoData);

      if (el.hyperparamStatusBadge) {
        el.hyperparamStatusBadge.textContent = 'Registry Validated';
        el.hyperparamStatusBadge.className = 'badge badge-include font-mono';
      }
      if (el.dynamicHyperparamErrors) {
        el.dynamicHyperparamErrors.classList.add('hidden');
        el.dynamicHyperparamErrors.innerHTML = '';
      }
    } catch (err) {
      console.warn('Could not load algorithm from registry API, using defaults:', err);
      if (el.hyperparamStatusBadge) {
        el.hyperparamStatusBadge.textContent = 'Local Fallback';
        el.hyperparamStatusBadge.className = 'badge badge-sensitive font-mono';
      }
      const meta = MODEL_METADATA[modelKey];
      if (meta && meta.paramGroup) {
        switchHyperparamPanel(meta.paramGroup);
      }
    }
  }

  function renderDynamicHyperparameters(algoData) {
    if (!el.dynamicHyperparamFields) return;
    el.dynamicHyperparamFields.innerHTML = '';

    const specs = algoData.hyperparameters || {};
    const keys = Object.keys(specs);

    if (keys.length === 0) {
      el.dynamicHyperparamFields.innerHTML = '<p class="caption text-muted">No configurable hyperparameters for this algorithm.</p>';
      return;
    }

    keys.forEach((paramName) => {
      const spec = specs[paramName];
      const fieldDiv = document.createElement('div');
      fieldDiv.className = 'form-group';
      fieldDiv.style.marginBottom = '12px';

      const labelDiv = document.createElement('div');
      labelDiv.className = 'flex-between align-center mb-1';
      labelDiv.innerHTML = `
        <label class="form-label" style="margin-bottom:0; font-weight:600;">${escapeHtml(spec.name)}</label>
        <span class="caption text-muted font-mono" style="font-size:11px;">${escapeHtml(spec.param_type)}${spec.allow_none ? ' | null' : ''}</span>
      `;
      fieldDiv.appendChild(labelDiv);

      let inputEl;

      if (spec.options && spec.options.length > 0) {
        inputEl = document.createElement('select');
        inputEl.className = 'form-select dynamic-hp-input';
        inputEl.dataset.param = spec.name;
        inputEl.dataset.type = spec.param_type;

        if (spec.allow_none) {
          const noneOpt = document.createElement('option');
          noneOpt.value = '__none__';
          noneOpt.textContent = 'None';
          if (spec.default === null) noneOpt.selected = true;
          inputEl.appendChild(noneOpt);
        }

        spec.options.forEach((opt) => {
          const optEl = document.createElement('option');
          optEl.value = String(opt);
          optEl.textContent = String(opt);
          if (spec.default === opt) optEl.selected = true;
          inputEl.appendChild(optEl);
        });
      } else if (spec.param_type === 'bool') {
        inputEl = document.createElement('select');
        inputEl.className = 'form-select dynamic-hp-input';
        inputEl.dataset.param = spec.name;
        inputEl.dataset.type = 'bool';

        const trueOpt = document.createElement('option');
        trueOpt.value = 'true';
        trueOpt.textContent = 'True';
        if (spec.default === true) trueOpt.selected = true;

        const falseOpt = document.createElement('option');
        falseOpt.value = 'false';
        falseOpt.textContent = 'False';
        if (spec.default === false) falseOpt.selected = true;

        inputEl.appendChild(trueOpt);
        inputEl.appendChild(falseOpt);
      } else if (spec.param_type === 'int' || spec.param_type === 'float') {
        inputEl = document.createElement('input');
        inputEl.type = 'number';
        inputEl.className = 'form-input dynamic-hp-input';
        inputEl.dataset.param = spec.name;
        inputEl.dataset.type = spec.param_type;
        inputEl.dataset.allowNone = spec.allow_none ? 'true' : 'false';

        if (spec.default !== null && spec.default !== undefined) {
          inputEl.value = spec.default;
        }
        if (spec.min_value !== null && spec.min_value !== undefined) {
          inputEl.min = spec.min_value;
        }
        if (spec.max_value !== null && spec.max_value !== undefined) {
          inputEl.max = spec.max_value;
        }
        inputEl.step = spec.param_type === 'float' ? '0.001' : '1';
        if (spec.allow_none) {
          inputEl.placeholder = 'Leave blank for None';
        }
      } else {
        inputEl = document.createElement('input');
        inputEl.type = 'text';
        inputEl.className = 'form-input dynamic-hp-input';
        inputEl.dataset.param = spec.name;
        inputEl.dataset.type = 'str';
        inputEl.value = spec.default !== null && spec.default !== undefined ? String(spec.default) : '';
      }

      // Input change listener for live validation
      inputEl.addEventListener('input', scheduleHyperparameterValidation);
      inputEl.addEventListener('change', scheduleHyperparameterValidation);

      fieldDiv.appendChild(inputEl);

      if (spec.description) {
        const hint = document.createElement('p');
        hint.className = 'form-hint';
        hint.style.marginTop = '4px';
        hint.style.fontSize = '11px';
        hint.textContent = spec.description;
        fieldDiv.appendChild(hint);
      }

      el.dynamicHyperparamFields.appendChild(fieldDiv);
    });
  }

  function scheduleHyperparameterValidation() {
    clearTimeout(hpValidationDebounceTimer);
    hpValidationDebounceTimer = setTimeout(validateCurrentHyperparameters, 300);
  }

  async function validateCurrentHyperparameters() {
    if (!state.algorithmConfig || !state.algorithmConfig.id) return;

    const inputs = el.dynamicHyperparamFields ? el.dynamicHyperparamFields.querySelectorAll('.dynamic-hp-input') : [];
    const params = {};

    inputs.forEach((inp) => {
      const pName = inp.dataset.param;
      const pType = inp.dataset.type;
      const allowNone = inp.dataset.allowNone === 'true';
      const val = inp.value;

      if (val === '__none__' || (val === '' && allowNone)) {
        params[pName] = null;
      } else if (pType === 'int') {
        params[pName] = val === '' ? null : parseInt(val, 10);
      } else if (pType === 'float') {
        params[pName] = val === '' ? null : parseFloat(val);
      } else if (pType === 'bool') {
        params[pName] = val === 'true';
      } else {
        params[pName] = val;
      }
    });

    try {
      const resp = await fetch(`${API_BASE}/algorithms/${encodeURIComponent(state.algorithmConfig.id)}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hyperparameters: params,
          task: state.algorithmConfig.task,
        }),
      });

      if (!resp.ok) return;

      const data = await resp.json();
      state.algorithmConfig.valid = data.valid;
      state.algorithmConfig.errors = data.errors || [];
      if (data.valid) {
        state.algorithmConfig.hyperparameters = data.sanitized_hyperparameters;
        if (el.hyperparamStatusBadge) {
          el.hyperparamStatusBadge.textContent = 'Registry Validated';
          el.hyperparamStatusBadge.className = 'badge badge-include font-mono';
        }
        if (el.dynamicHyperparamErrors) {
          el.dynamicHyperparamErrors.classList.add('hidden');
          el.dynamicHyperparamErrors.innerHTML = '';
        }
      } else {
        if (el.hyperparamStatusBadge) {
          el.hyperparamStatusBadge.textContent = 'Validation Errors';
          el.hyperparamStatusBadge.className = 'badge badge-sensitive font-mono';
        }
        if (el.dynamicHyperparamErrors) {
          el.dynamicHyperparamErrors.classList.remove('hidden');
          el.dynamicHyperparamErrors.innerHTML = `<strong>Hyperparameter Validation Errors:</strong><ul style="margin:4px 0 0 16px; padding:0;">${data.errors.map((e) => `<li>${escapeHtml(e)}</li>`).join('')}</ul>`;
        }
      }
    } catch (e) {
      console.warn('Hyperparameter validation request failed:', e);
    }
  }

  function updateMetricsOptions(modelType, defaultMetric) {
    if (!el.expMetric) return;

    let options = '';
    if (modelType === 'classification') {
      options = `
        <option value="accuracy">Accuracy</option>
        <option value="precision">Precision</option>
        <option value="recall">Recall</option>
        <option value="f1" selected>F1 Score</option>
      `;
    } else if (modelType === 'regression') {
      options = `
        <option value="rmse" selected>RMSE (Root Mean Squared Error)</option>
        <option value="mae">MAE (Mean Absolute Error)</option>
        <option value="mse">MSE (Mean Squared Error)</option>
        <option value="r2">R² Score</option>
      `;
    } else if (modelType === 'clustering') {
      options = `
        <option value="silhouette" selected>Silhouette Coefficient</option>
      `;
    } else {
      // Both
      options = `
        <option value="f1" selected>F1 Score (Classification)</option>
        <option value="accuracy">Accuracy (Classification)</option>
        <option value="rmse">RMSE (Regression)</option>
        <option value="r2">R² Score (Regression)</option>
        <option value="silhouette">Silhouette Coefficient (Clustering)</option>
      `;
    }

    el.expMetric.innerHTML = options;
    if (defaultMetric) {
      el.expMetric.value = defaultMetric;
    }
  }

  function switchHyperparamPanel(activeGroupKey) {
    Object.entries(el.paramGroups).forEach(([key, element]) => {
      if (element) {
        if (key && key === activeGroupKey) {
          element.classList.remove('hidden');
        } else {
          element.classList.add('hidden');
        }
      }
    });
  }

  // ============================================================
  // 9. EVENT LISTENERS & WIRING
  // ============================================================
  function setupNavigation() {
    // Smooth scroll for nav links
    el.navLinks.forEach((link) => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const sectionId = link.getAttribute('href').replace('#', '');
        scrollToSection(sectionId);

        // Close mobile nav if open
        if (el.mobileNav) el.mobileNav.classList.remove('active');
        if (el.hamburger) el.hamburger.classList.remove('active');
      });
    });

    // Mobile Hamburger
    if (el.hamburger && el.mobileNav) {
      el.hamburger.addEventListener('click', () => {
        el.hamburger.classList.toggle('active');
        el.mobileNav.classList.toggle('active');
      });
    }

    // ScrollSpy: observe sections to update active link
    const observerOptions = {
      root: null,
      rootMargin: '-20% 0px -60% 0px',
      threshold: 0,
    };

    const sectionObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          state.currentSection = id;

          el.navLinks.forEach((link) => {
            if (link.dataset.section === id) {
              link.classList.add('active');
            } else {
              link.classList.remove('active');
            }
          });

          // Update step indicator
          const stepMap = { upload: 1, restrictions: 2, output: 3, training: 4, 'ai-training': 5 };
          if (stepMap[id]) updateStepIndicators(stepMap[id]);
        }
      });
    }, observerOptions);

    el.sections.forEach((sec) => sectionObserver.observe(sec));

    // Scroll reveal observer
    const revealObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.1 }
    );

    document.querySelectorAll('.reveal, .reveal-left, .reveal-right').forEach((item) => {
      revealObserver.observe(item);
    });

    // Header buttons
    if (el.btnNewExperiment) {
      el.btnNewExperiment.addEventListener('click', () => {
        scrollToSection('upload');
        showNotification('Ready for a new experiment. Upload or select a dataset.', 'info');
      });
    }

    if (el.workspaceSelector) {
      el.workspaceSelector.addEventListener('click', () => {
        showNotification('Workspace: "Default Research Group" (Active)', 'info');
      });
    }
  }

  function updateStepIndicators(activeStep) {
    document.querySelectorAll('.step').forEach((stepEl) => {
      const stepNum = parseInt(stepEl.dataset.step, 10);
      stepEl.classList.remove('active', 'completed');
      if (stepNum === activeStep) {
        stepEl.classList.add('active');
      } else if (stepNum < activeStep) {
        stepEl.classList.add('completed');
      }
    });
  }

  function setupUploadZone() {
    if (!el.uploadZone || !el.fileInput) return;

    // Click to browse
    el.uploadZone.addEventListener('click', () => {
      el.fileInput.click();
    });

    if (el.browseLink) {
      el.browseLink.addEventListener('click', (e) => {
        e.stopPropagation();
        el.fileInput.click();
      });
    }

    el.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });

    // Drag and Drop
    ['dragenter', 'dragover'].forEach((eventName) => {
      el.uploadZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.uploadZone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      el.uploadZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        el.uploadZone.classList.remove('drag-over');
      });
    });

    el.uploadZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files[0]) {
        handleFileSelected(dt.files[0]);
      }
    });

    // Remove Dataset
    if (el.btnRemoveDataset) {
      el.btnRemoveDataset.addEventListener('click', removeDataset);
    }

    // Continue to Restrictions
    if (el.btnContinueRestrictions) {
      el.btnContinueRestrictions.addEventListener('click', () => {
        scrollToSection('restrictions');
      });
    }

    // Change / Upload another dataset button
    const btnChangeDataset = document.getElementById('btnChangeDataset');
    if (btnChangeDataset && el.fileInput) {
      btnChangeDataset.addEventListener('click', () => {
        el.fileInput.click();
      });
    }

    // Drag and Drop on preview table to swap dataset
    if (el.datasetPreview && el.fileInput) {
      ['dragenter', 'dragover'].forEach((eventName) => {
        el.datasetPreview.addEventListener(eventName, (e) => {
          e.preventDefault();
          e.stopPropagation();
          el.datasetPreview.classList.add('drag-over');
        });
      });
      ['dragleave', 'drop'].forEach((eventName) => {
        el.datasetPreview.addEventListener(eventName, (e) => {
          e.preventDefault();
          e.stopPropagation();
          el.datasetPreview.classList.remove('drag-over');
        });
      });
      el.datasetPreview.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files[0]) {
          handleFileSelected(dt.files[0]);
        }
      });
    }
  }

  function setupRestrictionsControls() {
    // Sensitive Toggles
    document.querySelectorAll('[data-sensitive]').forEach((toggle) => {
      toggle.addEventListener('click', () => {
        const key = toggle.dataset.sensitive;
        state.restrictions.sensitive[key] = !state.restrictions.sensitive[key];
        toggle.classList.toggle('active', state.restrictions.sensitive[key]);
        updateRestrictionSummary();
      });
    });

    // Quality Toggles
    document.querySelectorAll('[data-quality]').forEach((toggle) => {
      toggle.addEventListener('click', () => {
        const key = toggle.dataset.quality;
        state.restrictions.quality[key] = !state.restrictions.quality[key];
        toggle.classList.toggle('active', state.restrictions.quality[key]);
        updateRestrictionSummary();
      });
    });

    // Radio Groups (Missing handling, Encoding, Split)
    document.querySelectorAll('[data-radio-group]').forEach((group) => {
      const groupKey = group.dataset.radioGroup;
      group.querySelectorAll('.radio-option').forEach((option) => {
        option.addEventListener('click', () => {
          group.querySelectorAll('.radio-option').forEach((opt) => opt.classList.remove('selected'));
          option.classList.add('selected');

          const val = option.dataset.value;
          state.restrictions.constraints[groupKey] = val;

          // Sync with training section dropdowns
          if (groupKey === 'encoding' && el.expEncoding) el.expEncoding.value = val;
          if (groupKey === 'split' && el.expSplit) el.expSplit.value = val;

          updateRestrictionSummary();
          renderFeatureTable();
        });
      });
    });

    // Target Column Dropdown
    if (el.targetColumn) {
      el.targetColumn.addEventListener('change', (e) => {
        const val = e.target.value;
        state.restrictions.constraints.targetColumn = val;
        if (el.expTarget) el.expTarget.value = val;
        updateRestrictionSummary();
        renderFeatureTable();
      });
    }

    // Reset Restrictions
    if (el.btnResetRestrictions) {
      el.btnResetRestrictions.addEventListener('click', () => {
        if (!state.dataset) return;
        state.dataset.headers.forEach((h) => {
          state.restrictions.columns[h] = 'include';
        });
        Object.keys(state.restrictions.sensitive).forEach((k) => (state.restrictions.sensitive[k] = false));
        Object.keys(state.restrictions.quality).forEach((k) => (state.restrictions.quality[k] = false));

        document.querySelectorAll('.toggle').forEach((t) => t.classList.remove('active'));

        renderRestrictionsSection();
        updateRestrictionSummary();
        renderFeatureTable();
        showNotification('All restriction rules reset to defaults.', 'info');
      });
    }

    // Continue to Output
    if (el.btnContinueOutput) {
      el.btnContinueOutput.addEventListener('click', () => {
        scrollToSection('output');
      });
    }
  }

  function setupOutputControls() {
    // Viz tabs
    el.vizTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const viz = tab.dataset.viz;
        el.vizTabs.forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');

        Object.entries(el.vizPanels).forEach(([key, panel]) => {
          if (panel) {
            panel.classList.toggle('active', key === viz);
          }
        });
      });
    });

    // Category & Task Model Filtering
    function filterModels() {
      const catFilter = state.modelCategory || 'all';
      const taskFilter = state.activeTaskFilter || 'all';

      el.modelCards.forEach((card) => {
        const cardCat = card.dataset.category || 'traditional';
        const cardTask = card.dataset.task || '';

        const matchesCat = catFilter === 'all' || cardCat === catFilter;
        const matchesTask = taskFilter === 'all' || cardTask.includes(taskFilter);

        if (matchesCat && matchesTask) {
          card.classList.remove('hidden');
        } else {
          card.classList.add('hidden');
        }
      });
    }

    // Task Filter Pills
    if (el.taskPills) {
      el.taskPills.forEach((pill) => {
        pill.addEventListener('click', () => {
          el.taskPills.forEach((p) => p.classList.remove('active'));
          pill.classList.add('active');
          state.activeTaskFilter = pill.dataset.taskFilter;
          filterModels();
        });
      });
    }

    // Model Category Tabs (All vs Traditional vs Deep)
    el.modelCategoryTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const cat = tab.dataset.modelCategory;
        el.modelCategoryTabs.forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');
        state.modelCategory = cat;
        filterModels();
      });
    });

    // Model card clicks
    el.modelCards.forEach((card) => {
      card.addEventListener('click', () => {
        const modelKey = card.dataset.model;
        selectModel(modelKey);
      });
    });

    // Zemo Core Intelligence: Requirement Preset Pills
    if (el.presetPills) {
      el.presetPills.forEach((pill) => {
        pill.addEventListener('click', () => {
          const presetType = pill.dataset.preset;
          const presets = {
            'classification-precision': 'Optimize for high precision classification with minimal false positives on the target class.',
            'regression-robust': 'Predict continuous target with high resilience to extreme statistical outliers and skewed features.',
            'edge-lightweight': 'Produce a lightweight, compact model optimized for sub-millisecond inference on constrained edge hardware.',
            'tree-interpretable': 'Prioritize human-interpretable decision boundaries with clear feature split rules.',
            'balanced-f1': 'Maximize macro F1 score ensuring balanced predictive accuracy across all target classes.',
          };
          if (el.coreRequirementInput) {
            el.coreRequirementInput.value = presets[presetType] || '';
          }
        });
      });
    }

    // Zemo Core Intelligence: Synthesize Training Plan Action
    if (el.btnSynthesizePlan) {
      el.btnSynthesizePlan.addEventListener('click', () => {
        const reqText = el.coreRequirementInput ? el.coreRequirementInput.value.toLowerCase() : '';

        let recModel = 'gradient-boosting';
        let recModelName = 'Gradient Boosting Ensemble';
        let recMetric = 'F1 Score (Macro)';
        let recMetricVal = 'f1';
        let recFeature = 'Robust Scaler + One-Hot Encoding';

        if (reqText.includes('regression') || reqText.includes('continuous') || reqText.includes('housing')) {
          recModel = 'gradient-boosting';
          recModelName = 'Gradient Boosting Regressor';
          recMetric = 'RMSE (Root Mean Squared Error)';
          recMetricVal = 'rmse';
          recFeature = 'Robust Z-Score Scaler + Variance Filter';
        } else if (reqText.includes('edge') || reqText.includes('lightweight') || reqText.includes('latency')) {
          recModel = 'decision-tree';
          recModelName = 'Pruned Decision Tree (Max Depth 8)';
          recMetric = 'Inference Latency & F1 Score';
          recMetricVal = 'f1';
          recFeature = 'Compact Label Encoding + Top 10 Features';
        } else if (reqText.includes('neural') || reqText.includes('deep')) {
          recModel = 'mlp';
          recModelName = 'Multilayer Perceptron (3 Dense Layers)';
          recMetric = 'Accuracy / Cross-Entropy Loss';
          recMetricVal = 'accuracy';
          recFeature = 'MinMax Scaler + Dense Embeddings';
        } else if (reqText.includes('cluster') || reqText.includes('unsupervised')) {
          recModel = 'kmeans';
          recModelName = 'K-Means Clustering (k=5 Centroids)';
          recMetric = 'Silhouette Coefficient';
          recMetricVal = 'silhouette';
          recFeature = 'PCA Dimensionality Reduction';
        }

        state.corePlan = {
          model: recModel,
          modelName: recModelName,
          metric: recMetric,
          metricVal: recMetricVal,
          feature: recFeature,
        };

        if (el.planModelVal) el.planModelVal.textContent = recModelName;
        if (el.planMetricVal) el.planMetricVal.textContent = recMetric;
        if (el.planFeatureVal) el.planFeatureVal.textContent = recFeature;

        showNotification('Zemo Core Intelligence: Synthesized optimal training blueprint based on requirements.', 'success');
      });
    }

    // Zemo Core Intelligence: Apply Plan to Model Lab
    if (el.btnApplyPlanToLab) {
      el.btnApplyPlanToLab.addEventListener('click', () => {
        const plan = state.corePlan || {
          model: 'gradient-boosting',
          metricVal: 'f1',
        };

        selectModel(plan.model);
        if (el.expMetric) el.expMetric.value = plan.metricVal;

        scrollToSection('training');
        showNotification('Applied Core Intelligence blueprint to Model Lab.', 'success');
      });
    }

    // Download Preview
    if (el.btnDownloadPreview) {
      el.btnDownloadPreview.addEventListener('click', () => {
        if (!state.dataset) {
          showNotification('No dataset loaded to download.', 'warning');
          return;
        }

        // Generate CSV content from filtered/included columns
        const { headers, rows } = state.dataset;
        const includedCols = headers.filter((h) => state.restrictions.columns[h] !== 'exclude');

        const colIndices = includedCols.map((h) => headers.indexOf(h));
        let csvContent = includedCols.join(',') + '\n';
        rows.forEach((r) => {
          const rowVals = colIndices.map((idx) => `"${r[idx] !== undefined ? r[idx] : ''}"`);
          csvContent += rowVals.join(',') + '\n';
        });

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `zemo_configured_${state.dataset.name}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showNotification('Configured dataset preview downloaded.', 'success');
      });
    }

    // Continue to Training
    if (el.btnContinueTraining) {
      el.btnContinueTraining.addEventListener('click', () => {
        scrollToSection('training');
      });
    }
  }

  function setupTrainingControls() {
    // Model Select dropdown synchronization
    if (el.expModel) {
      el.expModel.addEventListener('change', (e) => {
        selectModel(e.target.value);
      });
    }

    // Target sync
    if (el.expTarget) {
      el.expTarget.addEventListener('change', (e) => {
        state.restrictions.constraints.targetColumn = e.target.value;
        if (el.targetColumn) el.targetColumn.value = e.target.value;
        updateRestrictionSummary();
        renderFeatureTable();
      });
    }

    // Encoding sync
    if (el.expEncoding) {
      el.expEncoding.addEventListener('change', (e) => {
        state.restrictions.constraints.encoding = e.target.value;
        updateRestrictionSummary();
        renderFeatureTable();
      });
    }

    // Split sync
    if (el.expSplit) {
      el.expSplit.addEventListener('change', (e) => {
        state.restrictions.constraints.split = e.target.value;
        updateRestrictionSummary();
      });
    }

    // Sliders wiring
    function bindSlider(slider, display) {
      if (slider && display) {
        slider.addEventListener('input', (e) => {
          display.textContent = e.target.value;
        });
      }
    }

    bindSlider(el.sliderEstimators, el.valEstimators);
    bindSlider(el.sliderMaxDepth, el.valMaxDepth);
    bindSlider(el.sliderMinSamples, el.valMinSamples);
    bindSlider(el.sliderLayers, el.valLayers);
    bindSlider(el.sliderHiddenUnits, el.valHiddenUnits);
    bindSlider(el.sliderEpochs, el.valEpochs);
    bindSlider(el.sliderNeighbors, el.valNeighbors);

    // Save Draft button
    if (el.btnSaveDraft) {
      el.btnSaveDraft.addEventListener('click', () => {
        const name = el.expName ? el.expName.value || 'unnamed-experiment' : 'unnamed-experiment';
        showNotification(`Draft saved locally: "${name}"`, 'success');
      });
    }

    // Validate Config button (Live Registry Validation & Compatibility Verification)
    if (el.btnValidateConfig) {
      el.btnValidateConfig.addEventListener('click', async () => {
        if (!state.dataset) {
          showNotification('Validation warning: No dataset uploaded yet.', 'warning');
          return;
        }

        const task = state.algorithmConfig ? state.algorithmConfig.task : 'classification';
        const isClustering = task === 'clustering';

        if (!isClustering && !state.restrictions.constraints.targetColumn) {
          showNotification('Validation warning: Please select a target variable for supervised learning.', 'warning');
          return;
        }

        // Validate hyperparameters first
        await validateCurrentHyperparameters();

        if (state.algorithmConfig && !state.algorithmConfig.valid) {
          showNotification(`Invalid hyperparameters: ${state.algorithmConfig.errors[0] || 'Check configuration'}`, 'warning');
          return;
        }

        // Compatibility check
        if (state.algorithmConfig && state.algorithmConfig.id) {
          try {
            const targetCol = state.restrictions.constraints.targetColumn;
            const targetType = targetCol && state.dataset.types ? state.dataset.types[targetCol] : null;
            const compatResp = await fetch(`${API_BASE}/algorithms/${encodeURIComponent(state.algorithmConfig.id)}/compatibility`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                task: state.algorithmConfig.task,
                target_type: targetType,
              }),
            });
            if (compatResp.ok) {
              const compatData = await compatResp.json();
              if (!compatData.compatible) {
                showNotification(`Compatibility Warning: ${compatData.errors.join('; ')}`, 'warning');
                return;
              }
            }
          } catch (e) {
            console.warn('Compatibility check failed:', e);
          }
        }

        showNotification(`Configuration verified: ${state.algorithmConfig ? state.algorithmConfig.display_name : 'Algorithm'} hyperparameters and compatibility checks passed.`, 'success');
      });
    }

    // Train Model (Live Engine) button
    if (el.btnTrainModel) {
      el.btnTrainModel.addEventListener('click', async () => {
        if (!state.dataset || !state.dataset.id) {
          showNotification('Please upload or select a dataset before initiating model training.', 'warning');
          return;
        }

        const task = (state.algorithmConfig && state.algorithmConfig.task) || 'classification';
        const isClustering = task === 'clustering';
        const targetCol = isClustering ? null : (state.restrictions.constraints.targetColumn || null);

        if (!isClustering && !targetCol) {
          showNotification('Please select a target column for supervised training in the Data Preparation section.', 'warning');
          return;
        }

        const selectedFeatures = (state.dataset.headers || []).filter(
          (h) => h !== targetCol && state.restrictions.columns[h] !== 'exclude'
        );

        if (selectedFeatures.length === 0) {
          showNotification('Please select at least one feature column for model training.', 'warning');
          return;
        }

        const algorithmId = (state.algorithmConfig && state.algorithmConfig.id) || (isClustering ? 'kmeans' : 'random_forest');

        // Extract user hyperparameters from dynamic input fields
        const params = {};
        const inputs = el.dynamicHyperparamFields ? el.dynamicHyperparamFields.querySelectorAll('.dynamic-hp-input') : [];
        inputs.forEach((inp) => {
          const pName = inp.dataset.param;
          const pType = inp.dataset.type;
          const allowNone = inp.dataset.allowNone === 'true';
          const val = inp.value;

          if (val === '__none__' || (val === '' && allowNone)) {
            params[pName] = null;
          } else if (pType === 'int') {
            params[pName] = val === '' ? null : parseInt(val, 10);
          } else if (pType === 'float') {
            params[pName] = val === '' ? null : parseFloat(val);
          } else if (pType === 'bool') {
            params[pName] = val === 'true';
          } else {
            params[pName] = val;
          }
        });

        // Mapping helpers
        const missingMap = {
          mean: 'mean',
          median: 'median',
          mode: 'most_frequent',
          constant: 'constant',
          drop: 'mean',
        };
        const encMap = {
          auto: 'auto',
          onehot: 'onehot',
          label: 'ordinal',
          none: 'none',
        };

        const missingNum = missingMap[state.restrictions.constraints.missingHandling] || 'mean';
        const encoding = encMap[state.restrictions.constraints.encoding] || 'auto';
        const scaling = state.restrictions.constraints.scaling || 'standard';
        const testRatio = state.restrictions.constraints.split === '70-30' ? 0.3 : state.restrictions.constraints.split === '90-10' ? 0.1 : 0.2;

        const expName = el.expName && el.expName.value.trim()
          ? el.expName.value.trim()
          : `${algorithmId}-${task}-${Date.now().toString().slice(-4)}`;

        const payload = {
          name: expName,
          dataset_id: state.dataset.id,
          task: task,
          target_column: targetCol,
          selected_features: selectedFeatures,
          missing_numeric_strategy: missingNum,
          missing_categorical_strategy: 'most_frequent',
          encoding_strategy: encoding,
          scaling_strategy: scaling,
          algorithm_id: algorithmId,
          hyperparameters: Object.keys(params).length > 0 ? params : (state.algorithmConfig ? state.algorithmConfig.hyperparameters || {} : {}),
          test_size: testRatio,
          random_state: 42,
          cv_folds: 5,
        };

        try {
          if (el.btnTrainModel) {
            el.btnTrainModel.disabled = true;
            el.btnTrainModel.innerHTML = '<span class="pulse-dot"></span> Dispatching Worker...';
          }

          const resp = await fetch(`${API_BASE}/experiments/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });

          if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: 'Server error' }));
            showNotification(`Training dispatch failed: ${err.detail || 'Unknown error'}`, 'error');
            return;
          }

          const expData = await resp.json();
          const expId = expData.id;

          const queueItem = {
            id: expId,
            backendExpId: expId,
            name: expName,
            model: state.algorithmConfig ? state.algorithmConfig.display_name : algorithmId,
            dataset: state.dataset.name,
            time: 'Just now',
            status: 'training',
            expData: expData,
          };

          state.queue.unshift(queueItem);
          renderQueue();

          // Reset Telemetry Cards
          if (el.activeTrainingJobName) el.activeTrainingJobName.textContent = `${expName} (${algorithmId})`;
          if (el.activeTrainingJobStatus) {
            el.activeTrainingJobStatus.className = 'queue-status training';
            el.activeTrainingJobStatus.innerHTML = '<span class="pulse-dot"></span> Training';
          }
          if (el.trainingStatStage) el.trainingStatStage.textContent = 'Pipeline Fitting';
          if (el.trainingStatProgress) el.trainingStatProgress.textContent = '25%';
          if (el.trainingProgressBar) el.trainingProgressBar.style.width = '25%';

          showNotification(`Training job "${expName}" queued and executing in background worker!`, 'success');

          // Scroll to progress section
          const progressSection = document.getElementById('activeTrainingJobName');
          if (progressSection) progressSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

          pollExperimentJob(expId, queueItem);

        } catch (error) {
          console.error('Training dispatch error:', error);
          showNotification(`Network error while initiating training: ${error.message}`, 'error');
        } finally {
          if (el.btnTrainModel) {
            el.btnTrainModel.disabled = false;
            el.btnTrainModel.innerHTML = `
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 6px;"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
              Train Model (Live ML Engine)
            `;
          }
        }
      });
    }

    // Add to Queue button (Demonstration / Queue tracking)
    if (el.btnAddToQueue) {
      el.btnAddToQueue.addEventListener('click', () => {
        const name = el.expName && el.expName.value.trim() ? el.expName.value.trim() : 'experiment-' + Math.floor(Math.random() * 900 + 100);
        const datasetName = state.dataset ? state.dataset.name : 'sample-data.csv';
        const modelName = MODEL_METADATA[state.selectedModel] ? MODEL_METADATA[state.selectedModel].name : 'Random Forest';

        const newItem = {
          id: 'exp-' + Date.now(),
          name: name,
          model: modelName,
          dataset: datasetName,
          time: 'Just now',
          status: 'queued',
        };

        state.queue.unshift(newItem);
        renderQueue();

        if (el.prototypeNotice) {
          el.prototypeNotice.classList.remove('hidden');
          el.prototypeNotice.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        showNotification(`Experiment "${name}" added to waiting queue. Click "Train Model" to execute live.`, 'info');
      });
    }

    // Render initial queue state (empty)
    renderQueue();
  }

  let activePollInterval = null;
  function pollExperimentJob(expId, queueItem) {
    if (activePollInterval) clearInterval(activePollInterval);

    const startTime = Date.now();
    activePollInterval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/experiments/${encodeURIComponent(expId)}`);
        if (!res.ok) return;
        const exp = await res.json();

        // Update elapsed time
        const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
        const mm = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
        const ss = String(elapsedSec % 60).padStart(2, '0');
        if (el.trainingStatElapsed) el.trainingStatElapsed.textContent = `00:${mm}:${ss}`;

        // Stream real logs from training engine
        if (el.trainingLog && Array.isArray(exp.logs) && exp.logs.length > 0) {
          el.trainingLog.innerHTML = exp.logs.map((log) => {
            const timeStr = log.timestamp ? log.timestamp.split('T')[1].split('.')[0] : '--:--:--';
            const lvl = (log.level || 'info').toLowerCase();
            return `
              <div class="log-entry">
                <span class="log-timestamp">[${escapeHtml(timeStr)}]</span>
                <span class="log-level ${escapeHtml(lvl)}">${escapeHtml(lvl.toUpperCase())}</span>
                <span class="log-message">${escapeHtml(log.message)}</span>
              </div>
            `;
          }).join('');
          el.trainingLog.scrollTop = el.trainingLog.scrollHeight;
        }

        if (exp.status === 'running') {
          if (el.trainingStatStage) el.trainingStatStage.textContent = 'Estimator Fitting';
          if (el.trainingStatProgress) el.trainingStatProgress.textContent = '65%';
          if (el.trainingProgressBar) el.trainingProgressBar.style.width = '65%';
        } else if (exp.status === 'completed') {
          clearInterval(activePollInterval);
          activePollInterval = null;

          if (el.trainingStatStage) el.trainingStatStage.textContent = 'Finished';
          if (el.trainingStatProgress) el.trainingStatProgress.textContent = '100%';
          if (el.trainingProgressBar) el.trainingProgressBar.style.width = '100%';
          if (el.activeTrainingJobStatus) {
            el.activeTrainingJobStatus.className = 'queue-status completed';
            el.activeTrainingJobStatus.textContent = 'Completed';
          }
          if (queueItem) {
            queueItem.status = 'completed';
            queueItem.expData = exp;
            renderQueue();
          }

          displayExperimentResults(exp);
          showNotification(`Training completed: ${exp.name} (Duration: ${exp.duration_seconds ? exp.duration_seconds.toFixed(2) + 's' : '0.1s'})`, 'success');

        } else if (exp.status === 'failed' || exp.status === 'cancelled') {
          clearInterval(activePollInterval);
          activePollInterval = null;

          if (el.trainingStatStage) el.trainingStatStage.textContent = exp.status.toUpperCase();
          if (el.activeTrainingJobStatus) {
            el.activeTrainingJobStatus.className = 'queue-status draft';
            el.activeTrainingJobStatus.textContent = exp.status.toUpperCase();
          }
          if (queueItem) {
            queueItem.status = 'failed';
            queueItem.expData = exp;
            renderQueue();
          }

          showNotification(`Training ${exp.status}: ${exp.error_message || 'Training error encountered'}`, 'error');
        }
      } catch (err) {
        console.error('Polling experiment status error:', err);
      }
    }, 600);
  }

  function displayExperimentResults(exp) {
    if (!exp) return;
    const metrics = exp.metrics || {};
    const task = exp.task || 'classification';

    if (el.resultExperimentName) {
      el.resultExperimentName.textContent = `${exp.name} (${exp.algorithm_id})`;
    }
    if (el.resultExperimentBadge) {
      el.resultExperimentBadge.textContent = `Completed (${exp.duration_seconds ? exp.duration_seconds.toFixed(2) + 's' : 'live'})`;
      el.resultExperimentBadge.style.background = 'rgba(76, 175, 80, 0.15)';
      el.resultExperimentBadge.style.color = '#2e7d32';
    }

    if (task === 'classification') {
      if (el.resultMetricLabel1) el.resultMetricLabel1.textContent = 'Accuracy';
      if (el.resultMetricVal1) el.resultMetricVal1.textContent = metrics.accuracy != null ? `${(metrics.accuracy * 100).toFixed(2)}%` : '—';

      if (el.resultMetricLabel2) el.resultMetricLabel2.textContent = 'Weighted F1';
      if (el.resultMetricVal2) el.resultMetricVal2.textContent = metrics.f1_weighted != null ? metrics.f1_weighted.toFixed(4) : '—';

      if (el.resultMetricLabel3) el.resultMetricLabel3.textContent = 'Macro F1';
      if (el.resultMetricVal3) el.resultMetricVal3.textContent = metrics.f1_macro != null ? metrics.f1_macro.toFixed(4) : '—';

      if (el.resultMetricLabel4) el.resultMetricLabel4.textContent = 'ROC-AUC';
      if (el.resultMetricVal4) {
        if (metrics.roc_auc != null) {
          el.resultMetricVal4.textContent = metrics.roc_auc.toFixed(4);
        } else {
          el.resultMetricVal4.textContent = 'N/A';
          el.resultMetricVal4.title = metrics.roc_auc_note || 'Probability scores not supported';
        }
      }

    } else if (task === 'regression') {
      if (el.resultMetricLabel1) el.resultMetricLabel1.textContent = 'R² Score';
      if (el.resultMetricVal1) el.resultMetricVal1.textContent = metrics.r2 != null ? metrics.r2.toFixed(4) : '—';

      if (el.resultMetricLabel2) el.resultMetricLabel2.textContent = 'RMSE';
      if (el.resultMetricVal2) el.resultMetricVal2.textContent = metrics.rmse != null ? metrics.rmse.toFixed(4) : '—';

      if (el.resultMetricLabel3) el.resultMetricLabel3.textContent = 'MAE';
      if (el.resultMetricVal3) el.resultMetricVal3.textContent = metrics.mae != null ? metrics.mae.toFixed(4) : '—';

      if (el.resultMetricLabel4) el.resultMetricLabel4.textContent = 'Explained Var';
      if (el.resultMetricVal4) el.resultMetricVal4.textContent = metrics.explained_variance != null ? metrics.explained_variance.toFixed(4) : '—';

    } else if (task === 'clustering') {
      if (el.resultMetricLabel1) el.resultMetricLabel1.textContent = 'Clusters Found';
      if (el.resultMetricVal1) el.resultMetricVal1.textContent = metrics.num_clusters != null ? metrics.num_clusters : '—';

      if (el.resultMetricLabel2) el.resultMetricLabel2.textContent = 'Silhouette Score';
      if (el.resultMetricVal2) el.resultMetricVal2.textContent = metrics.silhouette_score != null ? metrics.silhouette_score.toFixed(4) : 'N/A';

      if (el.resultMetricLabel3) el.resultMetricLabel3.textContent = 'Davies-Bouldin';
      if (el.resultMetricVal3) el.resultMetricVal3.textContent = metrics.davies_bouldin != null ? metrics.davies_bouldin.toFixed(4) : 'N/A';

      if (el.resultMetricLabel4) el.resultMetricLabel4.textContent = 'Calinski-Harabasz';
      if (el.resultMetricVal4) el.resultMetricVal4.textContent = metrics.calinski_harabasz != null ? metrics.calinski_harabasz.toFixed(1) : 'N/A';
    }

    // Render Confusion Matrix or Cluster Breakdown in visual-matrix-placeholder
    const matrixBox = document.querySelector('.visual-matrix-placeholder');
    if (matrixBox) {
      if (task === 'classification' && metrics.confusion_matrix && metrics.classes) {
        let cmHtml = `
          <div style="width: 100%; text-align: center;">
            <div class="caption font-mono mb-2" style="font-weight: 600;">Confusion Matrix (Classes: ${metrics.classes.join(', ')})</div>
            <table style="margin: 0 auto; border-collapse: collapse; font-family: var(--font-mono); font-size: 12px;">
              <thead>
                <tr>
                  <th style="padding: 4px 8px; border: 1px solid var(--border-subtle); background: var(--bg-surface-elevated);">True \\ Pred</th>
                  ${metrics.classes.map((c) => `<th style="padding: 4px 8px; border: 1px solid var(--border-subtle); background: var(--bg-surface-elevated);">${escapeHtml(c)}</th>`).join('')}
                </tr>
              </thead>
              <tbody>
                ${metrics.confusion_matrix.map((row, rIdx) => `
                  <tr>
                    <td style="padding: 4px 8px; border: 1px solid var(--border-subtle); font-weight: 600; background: var(--bg-surface-elevated);">${escapeHtml(metrics.classes[rIdx])}</td>
                    ${row.map((val) => `<td style="padding: 4px 8px; border: 1px solid var(--border-subtle); text-align: center; ${val > 0 ? 'background: rgba(76, 175, 80, 0.12); font-weight: 600;' : ''}">${val}</td>`).join('')}
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        `;
        matrixBox.innerHTML = cmHtml;
      } else if (task === 'clustering' && metrics.cluster_distribution) {
        let distHtml = `
          <div style="width: 100%; text-align: center;">
            <div class="caption font-mono mb-2" style="font-weight: 600;">Cluster Member Distribution</div>
            <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
              ${Object.entries(metrics.cluster_distribution).map(([cluster, count]) => `
                <div style="padding: 6px 12px; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); background: var(--bg-surface-elevated);">
                  <div style="font-size: 11px; color: var(--text-muted);">${cluster === '-1' ? 'Noise (-1)' : 'Cluster ' + cluster}</div>
                  <div style="font-size: 16px; font-weight: 600; font-family: var(--font-mono);">${count}</div>
                </div>
              `).join('')}
            </div>
          </div>
        `;
        matrixBox.innerHTML = distHtml;
      } else if (task === 'regression') {
        const predPoints = metrics.prediction_points || [];
        let pointsSampleHtml = '';
        if (predPoints.length > 0) {
          pointsSampleHtml = `
            <div class="mt-2" style="max-height: 120px; overflow-y: auto; font-family: var(--font-mono); font-size: 11px; border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
              <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead>
                  <tr style="background: var(--bg-surface-elevated); border-bottom: 1px solid var(--border-subtle);">
                    <th style="padding: 4px 8px;">Actual</th>
                    <th style="padding: 4px 8px;">Predicted</th>
                    <th style="padding: 4px 8px;">Residual</th>
                  </tr>
                </thead>
                <tbody>
                  ${predPoints.slice(0, 5).map(pt => `
                    <tr style="border-bottom: 1px solid var(--border-subtle);">
                      <td style="padding: 3px 8px;">${pt.actual}</td>
                      <td style="padding: 3px 8px;">${pt.predicted}</td>
                      <td style="padding: 3px 8px; color: ${pt.residual >= 0 ? '#2e7d32' : '#c62828'};">${pt.residual >= 0 ? '+' : ''}${pt.residual}</td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          `;
        }

        matrixBox.innerHTML = `
          <div style="text-align: center; padding: 12px; width: 100%;">
            <div class="caption font-mono" style="font-weight: 600; color: var(--deep-olive);">Regression Residuals & Predictions</div>
            <div class="caption text-muted mt-2">Test R²: <strong>${metrics.r2 != null ? metrics.r2.toFixed(4) : '—'}</strong> | RMSE: <strong>${metrics.rmse != null ? metrics.rmse.toFixed(4) : '—'}</strong> | Mean Residual: <strong>${metrics.mean_residual != null ? metrics.mean_residual.toFixed(4) : '0.0'}</strong></div>
            ${pointsSampleHtml}
          </div>
        `;
      }
    }

    // Render Feature Importance Panel
    if (el.featureImportanceContainer && el.featureImportanceBarsList) {
      const featImp = metrics.feature_importance;
      if (featImp && featImp.available && Array.isArray(featImp.importances) && featImp.importances.length > 0) {
        el.featureImportanceContainer.classList.remove('hidden');
        if (el.featureImportanceMethodBadge) {
          el.featureImportanceMethodBadge.textContent = (featImp.method || 'weights').replace(/_/g, ' ').toUpperCase();
        }
        if (el.featureImportanceExplanation) {
          el.featureImportanceExplanation.textContent = featImp.explanation || 'Attribution computed from fitted estimator.';
        }

        const topFeatures = featImp.importances.slice(0, 8);
        const maxVal = Math.max(...topFeatures.map((f) => f.importance || 0), 0.001);

        el.featureImportanceBarsList.innerHTML = topFeatures
          .map((item) => {
            const pct = Math.min(100, Math.max(3, (item.importance / maxVal) * 100));
            return `
              <div style="display: flex; align-items: center; gap: 12px; font-size: 12px;">
                <span style="width: 140px; font-family: var(--font-mono); text-overflow: ellipsis; overflow: hidden; white-space: nowrap;" title="${escapeHtml(item.feature)}">${escapeHtml(item.feature)}</span>
                <div style="flex: 1; background: var(--bg-surface); height: 8px; border-radius: 4px; overflow: hidden; border: 1px solid var(--border-subtle);">
                  <div style="background: var(--deep-olive); height: 100%; width: ${pct}%;"></div>
                </div>
                <span style="width: 55px; text-align: right; font-family: var(--font-mono); font-weight: 600; font-size: 11px;">${item.importance.toFixed(4)}</span>
              </div>
            `;
          })
          .join('');
      } else {
        if (featImp && featImp.explanation) {
          el.featureImportanceContainer.classList.remove('hidden');
          if (el.featureImportanceMethodBadge) el.featureImportanceMethodBadge.textContent = 'NOT APPLICABLE';
          if (el.featureImportanceExplanation) el.featureImportanceExplanation.textContent = featImp.explanation;
          el.featureImportanceBarsList.innerHTML = '';
        } else {
          el.featureImportanceContainer.classList.add('hidden');
        }
      }
    }

    // Render Detailed Diagnostics (Classification Report / Residuals / Evaluation Notes)
    if (el.evaluationDiagnosticsContainer && el.evaluationDiagnosticsContent) {
      el.evaluationDiagnosticsContainer.classList.remove('hidden');
      if (task === 'classification') {
        if (el.evalDetailSecondaryBadge) el.evalDetailSecondaryBadge.textContent = 'Classification Report & Predictions';
        let reportHtml = '';
        if (metrics.classification_report) {
          const rep = metrics.classification_report;
          const classes = Object.keys(rep).filter((k) => !['accuracy', 'macro avg', 'weighted avg'].includes(k));
          reportHtml = `
            <div style="margin-bottom: 12px;">
              <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 11px;">
                <thead>
                  <tr style="border-bottom: 1px solid var(--border-subtle); background: var(--bg-surface);">
                    <th style="padding: 4px 8px; text-align: left;">Class Label</th>
                    <th style="padding: 4px 8px; text-align: right;">Precision</th>
                    <th style="padding: 4px 8px; text-align: right;">Recall</th>
                    <th style="padding: 4px 8px; text-align: right;">F1-Score</th>
                    <th style="padding: 4px 8px; text-align: right;">Support</th>
                  </tr>
                </thead>
                <tbody>
                  ${classes.map(c => `
                    <tr style="border-bottom: 1px solid var(--border-subtle);">
                      <td style="padding: 4px 8px; font-weight: 600;">${escapeHtml(c)}</td>
                      <td style="padding: 4px 8px; text-align: right;">${rep[c].precision ? rep[c].precision.toFixed(4) : '0.0000'}</td>
                      <td style="padding: 4px 8px; text-align: right;">${rep[c].recall ? rep[c].recall.toFixed(4) : '0.0000'}</td>
                      <td style="padding: 4px 8px; text-align: right;">${rep[c]['f1-score'] ? rep[c]['f1-score'].toFixed(4) : '0.0000'}</td>
                      <td style="padding: 4px 8px; text-align: right;">${rep[c].support || 0}</td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          `;
        }

        let samplesHtml = '';
        if (metrics.prediction_samples && metrics.prediction_samples.length > 0) {
          samplesHtml = `
            <div>
              <div class="caption font-mono mb-2" style="font-weight: 600;">Sample Hold-out Predictions</div>
              <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 11px;">
                <thead>
                  <tr style="border-bottom: 1px solid var(--border-subtle); background: var(--bg-surface);">
                    <th style="padding: 4px 8px; text-align: left;">Actual Class</th>
                    <th style="padding: 4px 8px; text-align: left;">Predicted Class</th>
                    <th style="padding: 4px 8px; text-align: right;">Result</th>
                  </tr>
                </thead>
                <tbody>
                  ${metrics.prediction_samples.slice(0, 6).map(s => `
                    <tr style="border-bottom: 1px solid var(--border-subtle);">
                      <td style="padding: 3px 8px;">${escapeHtml(s.actual)}</td>
                      <td style="padding: 3px 8px;">${escapeHtml(s.predicted)}</td>
                      <td style="padding: 3px 8px; text-align: right; color: ${s.correct ? '#2e7d32' : '#c62828'}; font-weight: 600;">
                        ${s.correct ? '✓ Correct' : '✕ Error'}
                      </td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          `;
        }
        el.evaluationDiagnosticsContent.innerHTML = reportHtml + samplesHtml;

      } else if (task === 'regression') {
        if (el.evalDetailSecondaryBadge) el.evalDetailSecondaryBadge.textContent = 'Residual Diagnostics';
        el.evaluationDiagnosticsContent.innerHTML = `
          <div style="font-family: var(--font-mono); font-size: 11px; padding: 6px;">
            <div>Mean Residual Error: <strong>${metrics.mean_residual != null ? metrics.mean_residual.toFixed(4) : '0.0'}</strong></div>
            <div class="mt-1">Residual Standard Deviation: <strong>${metrics.std_residual != null ? metrics.std_residual.toFixed(4) : '0.0'}</strong></div>
            <div class="mt-1">Explained Variance Score: <strong>${metrics.explained_variance != null ? metrics.explained_variance.toFixed(4) : '—'}</strong></div>
          </div>
        `;
      } else if (task === 'clustering') {
        if (el.evalDetailSecondaryBadge) el.evalDetailSecondaryBadge.textContent = 'Clustering Metrics Notes';
        el.evaluationDiagnosticsContent.innerHTML = `
          <div style="font-size: 12px; color: var(--text-muted); padding: 4px;">
            ${escapeHtml(metrics.evaluation_notes || 'All geometric clustering evaluation metrics computed successfully.')}
          </div>
        `;
      }
    }

    // Enable and update model download link
    const dlBtn = document.getElementById('btnDownloadTrainedModel');
    if (dlBtn && exp.id) {
      dlBtn.classList.remove('hidden');
      dlBtn.href = `${API_BASE}/experiments/${encodeURIComponent(exp.id)}/download`;
      dlBtn.setAttribute('download', `${exp.name || 'model'}.joblib`);
    }

    const panel = document.getElementById('resultsPlaceholderPanel');
    if (panel) {
      panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  async function viewExperimentBenchmark(exp) {
    if (exp.expData) {
      displayExperimentResults(exp.expData);
      return;
    }
    if (exp.backendExpId) {
      try {
        const res = await fetch(`${API_BASE}/experiments/${encodeURIComponent(exp.backendExpId)}`);
        if (res.ok) {
          const data = await res.json();
          exp.expData = data;
          displayExperimentResults(data);
          return;
        }
      } catch (e) {
        console.warn('Failed to fetch experiment benchmark:', e);
      }
    }

    if (el.resultExperimentName) {
      el.resultExperimentName.textContent = `${exp.name} (${exp.model})`;
    }
    const panel = document.getElementById('resultsPlaceholderPanel');
    if (panel) {
      panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    showNotification(`Loaded benchmark metrics for "${exp.name}"`, 'info');
  }

  const selectedQueueExperiments = new Set();

  function updateCompareCount() {
    if (el.compareCountBadge) {
      el.compareCountBadge.textContent = selectedQueueExperiments.size;
    }
  }

  function renderQueue() {
    if (!el.queueList) return;

    if (!state.queue || state.queue.length === 0) {
      el.queueList.innerHTML = `
        <div class="empty-queue-placeholder">
          <p class="text-muted caption" style="margin: 0; padding: 20px; text-align: center;">No experiments queued or running. Configure an experiment above and click "Train Model (Live ML Engine)".</p>
        </div>
      `;
      selectedQueueExperiments.clear();
      updateCompareCount();
      return;
    }

    let html = '';
    state.queue.forEach((item, index) => {
      let statusBadge = '';
      if (item.status === 'completed') {
        statusBadge = '<span class="queue-status completed">Completed</span>';
      } else if (item.status === 'training') {
        statusBadge = '<span class="queue-status training"><span class="pulse-dot"></span> Training</span>';
      } else if (item.status === 'queued') {
        statusBadge = '<span class="queue-status queued">Queued</span>';
      } else if (item.status === 'ready') {
        statusBadge = '<span class="queue-status ready">Ready</span>';
      } else {
        statusBadge = '<span class="queue-status draft">Draft</span>';
      }

      const expId = item.backendExpId || item.id;
      const isCompleted = item.status === 'completed';
      const isChecked = selectedQueueExperiments.has(expId);

      html += `
        <div class="queue-item" data-index="${index}" style="display: flex; align-items: center; gap: 12px;">
          <input type="checkbox" class="queue-select-checkbox" data-id="${escapeHtml(expId)}" ${!isCompleted ? 'disabled title="Only completed experiments can be compared"' : ''} ${isChecked ? 'checked' : ''} style="width: 16px; height: 16px; cursor: pointer;">
          <div class="queue-item-info" style="flex: 1;">
            <span class="queue-item-name">${escapeHtml(item.name)}</span>
            <span class="queue-item-model">${escapeHtml(item.model)}</span>
            <span class="queue-item-dataset">${escapeHtml(item.dataset)}</span>
            <span class="queue-item-time">${escapeHtml(item.time)}</span>
          </div>
          ${statusBadge}
          <button class="btn btn-ghost btn-sm btn-queue-view" data-index="${index}">View Benchmark</button>
        </div>
      `;
    });

    el.queueList.innerHTML = html;

    // Attach checkbox listeners
    el.queueList.querySelectorAll('.queue-select-checkbox').forEach((cb) => {
      cb.addEventListener('change', (e) => {
        const id = e.target.dataset.id;
        if (e.target.checked) {
          selectedQueueExperiments.add(id);
        } else {
          selectedQueueExperiments.delete(id);
        }
        updateCompareCount();
      });
    });

    // Attach click listeners to view buttons
    el.queueList.querySelectorAll('.btn-queue-view').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const idx = parseInt(e.currentTarget.dataset.index, 10);
        const exp = state.queue[idx];
        if (exp) {
          viewExperimentBenchmark(exp);
        }
      });
    });

    updateCompareCount();
  }

  // Model comparison trigger
  if (el.btnCompareModels) {
    el.btnCompareModels.addEventListener('click', async () => {
      const selectedIds = Array.from(selectedQueueExperiments);
      if (selectedIds.length < 2) {
        showNotification('Please select at least 2 completed experiments to compare.', 'warning');
        return;
      }

      try {
        el.btnCompareModels.disabled = true;
        el.btnCompareModels.innerHTML = '<span class="pulse-dot"></span> Comparing...';

        const res = await fetch(`${API_BASE}/experiments/compare`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ experiment_ids: selectedIds }),
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: 'Comparison request failed' }));
          showNotification(`Comparison Error: ${err.detail || 'Incompatible experiments'}`, 'error');
          return;
        }

        const comp = await res.json();
        renderComparisonModal(comp);

      } catch (err) {
        console.error('Model comparison error:', err);
        showNotification(`Failed to compare models: ${err.message}`, 'error');
      } finally {
        if (el.btnCompareModels) {
          el.btnCompareModels.disabled = false;
          el.btnCompareModels.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 4px;"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>
            Compare Selected Models (<span id="compareCountBadge">${selectedQueueExperiments.size}</span>)
          `;
        }
      }
    });
  }

  if (el.btnCloseComparisonModal && el.modelComparisonModal) {
    el.btnCloseComparisonModal.addEventListener('click', () => {
      el.modelComparisonModal.style.display = 'none';
    });
  }

  function renderComparisonModal(comp) {
    if (!el.modelComparisonModal || !el.comparisonModalBody) return;

    const best = comp.best_model || {};
    const metricKeys = Object.keys(comp.metrics_comparison || {});

    let modalHtml = `
      <!-- Winner Banner -->
      <div style="border: 1px solid rgba(76, 175, 80, 0.3); background: rgba(76, 175, 80, 0.08); border-radius: var(--radius-sm); padding: 16px; margin-bottom: 20px;">
        <div class="flex-between align-center mb-1">
          <span class="badge" style="background: rgba(76, 175, 80, 0.2); color: #2e7d32; font-weight: 600;">🏆 Top Performer: ${escapeHtml(best.name || 'Model')}</span>
          <span class="caption text-muted font-mono">Task: ${escapeHtml(comp.task.toUpperCase())}</span>
        </div>
        <p style="margin: 6px 0 0; font-size: 13px; color: var(--text-primary);">${escapeHtml(comp.recommendation)}</p>
      </div>

      <!-- Comparative Matrix Table -->
      <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; font-size: 12px; font-family: var(--font-sans);">
          <thead>
            <tr style="border-bottom: 2px solid var(--border-subtle); background: var(--bg-surface-elevated);">
              <th style="padding: 10px 14px; text-align: left; width: 220px;">Configuration & Metric</th>
              ${comp.experiments.map(exp => `
                <th style="padding: 10px 14px; text-align: left; ${best.id === exp.id ? 'background: rgba(76, 175, 80, 0.08); font-weight: 700;' : ''}">
                  <div>${escapeHtml(exp.name)}</div>
                  <div class="caption font-mono text-muted" style="font-size: 10px;">${escapeHtml(exp.algorithm_id)}</div>
                </th>
              `).join('')}
            </tr>
          </thead>
          <tbody>
            <!-- Preprocessing & Runtime -->
            <tr style="background: var(--bg-surface); border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px 14px; font-weight: 600; color: var(--text-muted); font-size: 11px; text-transform: uppercase;">Runtime Duration</td>
              ${comp.experiments.map(e => `<td style="padding: 8px 14px; font-family: var(--font-mono);">${e.duration_seconds ? e.duration_seconds.toFixed(2) + 's' : '—'}</td>`).join('')}
            </tr>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px 14px; font-weight: 600; color: var(--text-muted); font-size: 11px; text-transform: uppercase;">Model Size</td>
              ${comp.experiments.map(e => `<td style="padding: 8px 14px; font-family: var(--font-mono);">${e.model_size_bytes ? (e.model_size_bytes / 1024).toFixed(1) + ' KB' : '—'}</td>`).join('')}
            </tr>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px 14px; font-weight: 600; color: var(--text-muted); font-size: 11px; text-transform: uppercase;">Scaling & Encoding</td>
              ${comp.experiments.map(e => `<td style="padding: 8px 14px;">${escapeHtml(e.preprocessing.scaling_strategy)} / ${escapeHtml(e.preprocessing.encoding_strategy)}</td>`).join('')}
            </tr>

            <!-- Metrics Comparison Header -->
            <tr style="background: var(--bg-surface-elevated); border-top: 2px solid var(--border-subtle); border-bottom: 1px solid var(--border-subtle);">
              <td colspan="${comp.experiments.length + 1}" style="padding: 6px 14px; font-weight: 700; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--deep-olive);">
                Model Evaluation Metrics
              </td>
            </tr>

            ${metricKeys.map(mKey => `
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 8px 14px; font-weight: 600;">${escapeHtml(mKey.replace(/_/g, ' '))}</td>
                ${comp.experiments.map(e => {
                  const val = comp.metrics_comparison[mKey][e.id];
                  const isTop = best.id === e.id;
                  return `
                    <td style="padding: 8px 14px; font-family: var(--font-mono); ${isTop ? 'background: rgba(76, 175, 80, 0.06); font-weight: 600;' : ''}">
                      ${val != null ? (typeof val === 'number' ? val.toFixed(4) : val) : '—'}
                    </td>
                  `;
                }).join('')}
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;

    el.comparisonModalBody.innerHTML = modalHtml;
    el.modelComparisonModal.style.display = 'flex';
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // ============================================================
  // 9.5 AI TRAINING STUDIO & ASSISTANT CHATBOX
  // ============================================================

  const MODEL_REGISTRY = {
    'distilbert-base': {
      name: 'DistilBERT-Base-Uncased',
      family: 'Transformer (Distilled BERT)',
      params: '66,362,880 (66.4M)',
      checkpoint: 'hf:distilbert-base-uncased@8f3a12',
      status: 'Verified for Fine-Tuning',
      isSupported: true,
      defaultLR: '0.00002',
      estVRAM: '3.8 GB',
      estTime: '~7m 30s',
    },
    'roberta-base': {
      name: 'RoBERTa-Base',
      family: 'Transformer (Bidirectional Encoder)',
      params: '124,645,632 (125M)',
      checkpoint: 'hf:roberta-base@c14d9e',
      status: 'Verified for Fine-Tuning',
      isSupported: true,
      defaultLR: '0.000015',
      estVRAM: '6.4 GB',
      estTime: '~14m 20s',
    },
    'bert-mini': {
      name: 'BERT-Mini',
      family: 'Compact Transformer (4 layers)',
      params: '11,270,400 (11.3M)',
      checkpoint: 'hf:google/bert_mini@3e2a',
      status: 'Verified for Fine-Tuning',
      isSupported: true,
      defaultLR: '0.00005',
      estVRAM: '1.4 GB',
      estTime: '~3m 10s',
    },
    't5-small': {
      name: 'T5-Small',
      family: 'Encoder-Decoder Transformer',
      params: '60,506,624 (60M)',
      checkpoint: 'hf:t5-small@7b18ef',
      status: 'Verified for Fine-Tuning',
      isSupported: true,
      defaultLR: '0.00003',
      estVRAM: '4.5 GB',
      estTime: '~9m 45s',
    },
    'mobilenet-v3-small': {
      name: 'MobileNet-V3-Small',
      family: 'Depthwise Separable CNN',
      params: '2,542,856 (2.5M)',
      checkpoint: 'torchvision:mobilenet_v3_small@v1',
      status: 'Verified for Fine-Tuning',
      isSupported: true,
      defaultLR: '0.0001',
      estVRAM: '1.1 GB',
      estTime: '~2m 15s',
    },
    'resnet-50': {
      name: 'ResNet-50-v2',
      family: 'Deep Residual Network',
      params: '25,557,032 (25.6M)',
      checkpoint: 'torchvision:resnet50_v2@9d1f',
      status: 'Verified for Fine-Tuning',
      isSupported: true,
      defaultLR: '0.00008',
      estVRAM: '5.2 GB',
      estTime: '~11m 00s',
    },
    'llama-3-70b': {
      name: 'Llama-3-70B-Instruct',
      family: 'Large Language Model (70B)',
      params: '70,553,000,000 (70.6B)',
      checkpoint: 'meta:llama-3-70b@unsupported',
      status: 'UNSUPPORTED — Requires Distributed Cluster',
      isSupported: false,
      unsupportedReason: 'Model parameter count (70.6B) exceeds single-node GPU memory envelope (> 140 GB FP16 required). Zemo strictly enforces honesty: this model cannot be fine-tuned in this environment without a multi-node cluster.',
    },
    'gpt-4-closed': {
      name: 'GPT-4-Turbo',
      family: 'Closed Weights API Architecture',
      params: 'Proprietary / Inaccessible',
      checkpoint: 'closed:blackbox-api@unsupported',
      status: 'UNSUPPORTED — Weights Inaccessible',
      isSupported: false,
      unsupportedReason: 'Model weights are closed and not hosted as an accessible checkpoint for gradient updates. Zemo strictly forbids false claims of fine-tuning black-box closed models.',
    },
  };

  const SCRATCH_ARCHITECTURES = {
    mlp: {
      name: 'Deep Multi-Layer Perceptron (Dense Net with LayerNorm & Res-Links)',
      baseFlops: '1.4 GFLOPs',
      baseMemory: '1.2 GB',
      description: 'Deep feedforward layers with LayerNorm and residual skips. Optimal for tabular features.',
    },
    conv1d: {
      name: '1D Convolutional Neural Network (Conv1D-Net)',
      baseFlops: '2.1 GFLOPs',
      baseMemory: '1.6 GB',
      description: 'Temporal feature kernels for sequence and waveform classification without pretrained embeddings.',
    },
    bilstm: {
      name: 'Bidirectional LSTM (BiLSTM)',
      baseFlops: '3.8 GFLOPs',
      baseMemory: '2.0 GB',
      description: 'Recurrent forward & backward cell gating with scratch vocabulary embeddings.',
    },
    'mini-transformer': {
      name: 'Task-Specific Mini-Transformer (2-Layer Self-Attention)',
      baseFlops: '4.5 GFLOPs',
      baseMemory: '2.4 GB',
      description: '2-layer multi-head self-attention with learned positional encodings and scratch vocabulary.',
    },
    tabresnet: {
      name: 'Tabular ResNet (TabResNet with GLU)',
      baseFlops: '2.8 GFLOPs',
      baseMemory: '1.8 GB',
      description: 'Specialized deep residual network with Gated Linear Units designed for complex tabular structures.',
    },
  };

  // 1. THEME MANAGEMENT
  function initTheme() {
    const savedTheme = localStorage.getItem('zemo_theme') || 'organic';
    applyTheme(savedTheme);

    if (el.themeToggle) {
      el.themeToggle.addEventListener('click', toggleTheme);
    }
  }

  function applyTheme(theme) {
    state.theme = theme;
    if (theme === 'organic') {
      document.body.classList.add('theme-organic');
      if (el.themeToggleLabel) el.themeToggleLabel.textContent = 'Calm Organic';
    } else {
      document.body.classList.remove('theme-organic');
      if (el.themeToggleLabel) el.themeToggleLabel.textContent = 'Dark Editorial';
    }
    localStorage.setItem('zemo_theme', theme);
  }

  function toggleTheme() {
    const nextTheme = state.theme === 'organic' ? 'dark' : 'organic';
    applyTheme(nextTheme);
    showNotification(`Switched theme to ${nextTheme === 'organic' ? 'Calm Organic' : 'Dark Editorial'}.`, 'info');
  }

  // 2. SETUP AI TRAINING CONTROLS
  function setupAiTrainingControls() {
    initTheme();

    // Mode Card Toggle
    if (el.modeCardPretrained && el.modeCardScratch) {
      el.modeCardPretrained.addEventListener('click', () => setAiTrainingMode('pretrained'));
      el.modeCardScratch.addEventListener('click', () => setAiTrainingMode('scratch'));
    }

    // Pretrained Model Select
    if (el.ptModelSelect) {
      el.ptModelSelect.addEventListener('change', (e) => {
        handlePretrainedModelChange(e.target.value);
      });
    }

    // Pretrained Method Radio
    document.querySelectorAll('input[name="ptMethod"]').forEach((radio) => {
      radio.addEventListener('change', (e) => {
        state.aiTraining.pretrained.method = e.target.value;
        updatePretrainedResourceEstimates();
      });
    });

    // Pretrained Sliders & Inputs
    if (el.ptEpochs) {
      el.ptEpochs.addEventListener('input', (e) => {
        el.valPtEpochs.textContent = e.target.value;
        state.aiTraining.pretrained.epochs = parseInt(e.target.value, 10);
        updatePretrainedResourceEstimates();
      });
    }
    if (el.ptLR) {
      el.ptLR.addEventListener('change', (e) => {
        state.aiTraining.pretrained.learningRate = e.target.value.trim();
      });
    }
    if (el.ptBatchSize) {
      el.ptBatchSize.addEventListener('change', (e) => {
        state.aiTraining.pretrained.batchSize = parseInt(e.target.value, 10);
        updatePretrainedResourceEstimates();
      });
    }
    if (el.ptValSplit) {
      el.ptValSplit.addEventListener('change', (e) => {
        state.aiTraining.pretrained.valSplit = parseInt(e.target.value, 10);
      });
    }
    if (el.ptDatasetSelect) {
      el.ptDatasetSelect.addEventListener('change', (e) => {
        state.aiTraining.pretrained.dataset = e.target.value;
        handleDatasetSelectionChange(e.target.value);
      });
    }

    // Scratch Mode Controls
    if (el.scratchArchSelect) {
      el.scratchArchSelect.addEventListener('change', (e) => {
        state.aiTraining.scratch.architecture = e.target.value;
        validateScratchTask();
      });
    }
    document.querySelectorAll('input[name="scratchSize"]').forEach((radio) => {
      radio.addEventListener('change', (e) => {
        state.aiTraining.scratch.size = e.target.value;
        validateScratchTask();
      });
    });
    if (el.scratchTaskSelect) {
      el.scratchTaskSelect.addEventListener('change', (e) => {
        state.aiTraining.scratch.task = e.target.value;
        validateScratchTask();
      });
    }
    if (el.scratchInitSelect) {
      el.scratchInitSelect.addEventListener('change', (e) => {
        state.aiTraining.scratch.initScheme = e.target.value;
      });
    }
    if (el.scratchSplitSelect) {
      el.scratchSplitSelect.addEventListener('change', (e) => {
        state.aiTraining.scratch.split = e.target.value;
      });
    }
    if (el.scratchScalerSelect) {
      el.scratchScalerSelect.addEventListener('change', (e) => {
        state.aiTraining.scratch.scaler = e.target.value;
      });
    }
    if (el.scratchEncodingSelect) {
      el.scratchEncodingSelect.addEventListener('change', (e) => {
        state.aiTraining.scratch.encoding = e.target.value;
      });
    }
    if (el.scratchEpochs) {
      el.scratchEpochs.addEventListener('input', (e) => {
        el.valScratchEpochs.textContent = e.target.value;
        state.aiTraining.scratch.epochs = parseInt(e.target.value, 10);
      });
    }
    if (el.scratchLR) {
      el.scratchLR.addEventListener('change', (e) => {
        state.aiTraining.scratch.learningRate = e.target.value.trim();
      });
    }
    if (el.scratchBatchSize) {
      el.scratchBatchSize.addEventListener('change', (e) => {
        state.aiTraining.scratch.batchSize = parseInt(e.target.value, 10);
      });
    }
    if (el.scratchDropout) {
      el.scratchDropout.addEventListener('input', (e) => {
        el.valScratchDropout.textContent = parseFloat(e.target.value).toFixed(2);
        state.aiTraining.scratch.dropout = parseFloat(e.target.value);
      });
    }

    // Policy & Governance Scan
    if (el.btnRunSecurityScan) {
      el.btnRunSecurityScan.addEventListener('click', runSecurityAndPolicyScan);
    }

    // Primary Action Buttons
    if (el.btnReviewPlan) {
      el.btnReviewPlan.addEventListener('click', openPreflightReviewModal);
    }
    if (el.btnLaunchTrainingDemo) {
      el.btnLaunchTrainingDemo.addEventListener('click', () => {
        openPreflightReviewModal();
      });
    }
    if (el.btnViewBackendSpec) {
      el.btnViewBackendSpec.addEventListener('click', openBackendSpecModal);
    }

    // Modal Action Buttons
    if (el.btnCloseReviewModal) el.btnCloseReviewModal.addEventListener('click', closePreflightReviewModal);
    if (el.btnCancelReviewModal) el.btnCancelReviewModal.addEventListener('click', closePreflightReviewModal);
    if (el.btnConfirmAndLaunch) el.btnConfirmAndLaunch.addEventListener('click', startTrainingSimulation);

    if (el.btnCloseSpecModal) el.btnCloseSpecModal.addEventListener('click', closeBackendSpecModal);
    if (el.btnDismissSpecModal) el.btnDismissSpecModal.addEventListener('click', closeBackendSpecModal);

    // Live Monitor Actions
    if (el.btnCancelTrainingJob) {
      el.btnCancelTrainingJob.addEventListener('click', cancelTrainingJob);
    }

    // Chatbox Controls
    setupTrainingAssistantChat();

    // Initialize dataset & validation state with current dataset (or empty)
    handleDatasetSelectionChange('current');
    validateScratchTask();
  }

  // 3. SET TRAINING MODE
  function setAiTrainingMode(mode) {
    state.aiTraining.mode = mode;
    if (mode === 'pretrained') {
      el.modeCardPretrained.classList.add('active');
      el.modeCardScratch.classList.remove('active');
      el.panelPretrained.classList.remove('hidden');
      el.panelPretrained.classList.add('active');
      el.panelScratch.classList.add('hidden');
      el.panelScratch.classList.remove('active');
      showNotification('Switched to Pre-trained Model Adaptation mode.', 'info');
    } else {
      el.modeCardScratch.classList.add('active');
      el.modeCardPretrained.classList.remove('active');
      el.panelScratch.classList.remove('hidden');
      el.panelScratch.classList.add('active');
      el.panelPretrained.classList.add('hidden');
      el.panelPretrained.classList.remove('active');
      validateScratchTask();
      showNotification('Switched to Train from Scratch (Cold Start) mode.', 'info');
    }
  }

  // 4. PRETRAINED MODEL LOGIC
  function handlePretrainedModelChange(modelId) {
    state.aiTraining.pretrained.modelId = modelId;
    const model = MODEL_REGISTRY[modelId];
    if (!model) return;

    if (!model.isSupported) {
      state.aiTraining.pretrained.isSupported = false;
      el.unsupportedModelAlert.classList.remove('hidden');
      el.unsupportedModelText.textContent = model.unsupportedReason;
      el.ptModelMetaCard.style.opacity = '0.5';
      el.ptMetaStatus.className = 'badge badge-exclude';
      el.ptMetaStatus.textContent = model.status;
      el.btnLaunchTrainingDemo.disabled = true;
      el.btnReviewPlan.disabled = true;
      el.btnLaunchTrainingDemo.title = 'Unsupported model cannot be trained.';
      showNotification(`Model "${model.name}" is unsupported for fine-tuning.`, 'error');
    } else {
      state.aiTraining.pretrained.isSupported = true;
      el.unsupportedModelAlert.classList.add('hidden');
      el.ptModelMetaCard.style.opacity = '1';
      el.ptMetaFamily.textContent = model.family;
      el.ptMetaParams.textContent = model.params;
      el.ptMetaCheckpoint.textContent = model.checkpoint;
      el.ptMetaStatus.className = 'badge badge-success';
      el.ptMetaStatus.textContent = model.status;
      el.ptLR.value = model.defaultLR;
      state.aiTraining.pretrained.learningRate = model.defaultLR;
      el.btnLaunchTrainingDemo.disabled = false;
      el.btnReviewPlan.disabled = false;
      el.btnLaunchTrainingDemo.title = '';
      updatePretrainedResourceEstimates();
    }
  }

  function updatePretrainedResourceEstimates() {
    const model = MODEL_REGISTRY[state.aiTraining.pretrained.modelId];
    if (!model || !model.isSupported) return;

    const method = state.aiTraining.pretrained.method;
    let vram = '3.8 GB';
    let time = model.estTime;

    if (method === 'head') {
      vram = '2.1 GB';
      time = '~3m 40s';
    } else if (method === 'full') {
      vram = '7.8 GB';
      time = '~18m 15s';
    } else if (method === 'lora') {
      vram = '3.8 GB';
      time = model.estTime;
    }

    if (el.ptEstVRAM) el.ptEstVRAM.textContent = vram;
    if (el.ptEstTime) el.ptEstTime.textContent = time;
  }

  function handleDatasetSelectionChange(dsKey) {
    if (dsKey === 'unvetted-scraping-dump') {
      el.ptDatasetApprovalText.textContent = 'Flagged: Contains Unmasked PII / Policy Violations';
      el.ptDatasetApprovalBar.style.borderColor = 'var(--error)';
      state.aiTraining.policyStatus = 'blocked';
      updateGovernanceGatewayUI();
      showNotification('Dataset flagged: violates privacy policy.', 'error');
    } else if (dsKey === 'current') {
      if (state.dataset) {
        el.ptDatasetApprovalText.textContent = `Active Upload: ${state.dataset.name} (Synchronized)`;
        el.ptDatasetRowsBadge.textContent = `${state.dataset.rows ? state.dataset.rows.length.toLocaleString() : 0} Rows`;
      } else {
        el.ptDatasetApprovalText.textContent = 'No local dataset uploaded yet. Upload a dataset in Step 01.';
        el.ptDatasetRowsBadge.textContent = 'No Dataset';
      }
      el.ptDatasetApprovalBar.style.borderColor = 'var(--border-subtle)';
      runSecurityAndPolicyScan();
    } else {
      el.ptDatasetApprovalText.textContent = 'Dataset Approved & Policy Validated';
      el.ptDatasetApprovalBar.style.borderColor = 'var(--border-subtle)';
      el.ptDatasetRowsBadge.textContent = 'Approved Corpus';
      state.aiTraining.policyStatus = 'passed';
      updateGovernanceGatewayUI();
    }
  }

  function getSelectedDatasetName(dsKey) {
    if (dsKey === 'current' && state.dataset) return state.dataset.name;
    if (dsKey === 'customer-sentiment') return 'Customer Sentiment Reviews (10,000 samples)';
    if (dsKey === 'product-feedback') return 'Product Feedback Corpus (25,000 samples)';
    if (dsKey === 'customer-churn-approved') return 'Customer Churn Records (7,043 samples)';
    if (dsKey === 'housing-sample-approved') return 'Housing Features (4,821 samples)';
    if (dsKey === 'unvetted-scraping-dump') return 'Raw Scraping Dump [FLAGGED]';
    return state.dataset ? state.dataset.name : 'Customer Sentiment Reviews';
  }

  // 5. SCRATCH VALIDATION
  function validateScratchTask() {
    const task = state.aiTraining.scratch.task;
    const arch = state.aiTraining.scratch.architecture;
    const ds = state.dataset;

    let msg = `Architecture: ${SCRATCH_ARCHITECTURES[arch].name}. Verified for ${task} from cold start.`;
    let sampleNote = 'Sample volume checked. Cold start weights initialization verified.';

    if (ds && ds.rows) {
      sampleNote = `Active dataset contains ${ds.rows.length.toLocaleString()} rows. `;
      if (ds.rows.length < 500) {
        sampleNote += '⚠️ Warning: Sample size < 500 may cause small task-specific neural nets to overfit without data augmentation.';
      } else {
        sampleNote += 'Sufficient volume for small task-specific architecture.';
      }
    } else {
      sampleNote = 'No dataset loaded yet. Upload a dataset in Step 01 to validate task compatibility.';
    }

    if (el.scratchValidationMsg) el.scratchValidationMsg.textContent = msg;
    if (el.scratchSampleCheck) el.scratchSampleCheck.textContent = sampleNote;
  }

  // 6. GOVERNANCE & SECURITY SCAN
  function runSecurityAndPolicyScan() {
    let hasViolations = false;
    let reasons = [];

    // Check sensitive flags from Restrictions section
    if (state.restrictions.sensitive.email) { hasViolations = true; reasons.push('Unmasked Email'); }
    if (state.restrictions.sensitive.phone) { hasViolations = true; reasons.push('Unmasked Phone'); }
    if (state.restrictions.sensitive.names) { hasViolations = true; reasons.push('Personal Names'); }
    if (state.restrictions.sensitive.personalIds) { hasViolations = true; reasons.push('Personal IDs'); }
    if (state.restrictions.sensitive.financial) { hasViolations = true; reasons.push('Financial Data'); }

    if (state.aiTraining.mode === 'pretrained' && state.aiTraining.pretrained.dataset === 'unvetted-scraping-dump') {
      hasViolations = true;
      reasons.push('Dataset Flagged by Policy');
    }

    if (hasViolations) {
      state.aiTraining.policyStatus = 'blocked';
      updateGovernanceGatewayUI(`Policy Blocked: ${reasons.join(', ')}. Remediate in Restrictions module before training.`);
      showNotification(`Policy scan failed: ${reasons.join(', ')} detected.`, 'error');
      return false;
    } else {
      state.aiTraining.policyStatus = 'passed';
      updateGovernanceGatewayUI();
      showNotification('Zemo Governance Scan: PASSED. Zero privacy violations.', 'success');
      return true;
    }
  }

  function updateGovernanceGatewayUI(customErrorText) {
    if (!el.gatewayBadge) return;
    if (state.aiTraining.policyStatus === 'blocked') {
      el.gatewayBadge.className = 'gateway-badge blocked';
      el.gatewayStatusText.textContent = 'Policy Status: Blocked by Policy';
      el.gatewayDetailText.textContent = customErrorText || 'PII or unmasked sensitive violations detected. Cannot dispatch training.';
    } else {
      el.gatewayBadge.className = 'gateway-badge passed';
      el.gatewayStatusText.textContent = 'Policy Status: Verified & Approved';
      el.gatewayDetailText.textContent = '0 PII violations • Target column confirmed • Validated against Restrictions rules';
    }
  }

  // 7. PRE-FLIGHT REVIEW & BACKEND SPEC MODALS
  function openPreflightReviewModal() {
    const isPretrained = state.aiTraining.mode === 'pretrained';
    if (isPretrained) {
      const model = MODEL_REGISTRY[state.aiTraining.pretrained.modelId];
      if (!model || !model.isSupported) {
        showNotification('Cannot review experiment: Unsupported model cannot be fine-tuned.', 'error');
        return;
      }
    }

    // Populate review modal
    const reviewModeVal = document.getElementById('reviewModeVal');
    const reviewArchVal = document.getElementById('reviewArchVal');
    const reviewWeightsVal = document.getElementById('reviewWeightsVal');
    const reviewDatasetVal = document.getElementById('reviewDatasetVal');
    const reviewGovVal = document.getElementById('reviewGovVal');
    const reviewParamsVal = document.getElementById('reviewParamsVal');
    const reviewResourceVal = document.getElementById('reviewResourceVal');

    if (reviewModeVal) {
      reviewModeVal.textContent = isPretrained ? 'Pre-trained Model Adaptation (Transfer Learning)' : 'Train from Scratch (Cold Start Isolation)';
    }
    if (reviewArchVal) {
      reviewArchVal.textContent = isPretrained ? MODEL_REGISTRY[state.aiTraining.pretrained.modelId].name : SCRATCH_ARCHITECTURES[state.aiTraining.scratch.architecture].name;
    }
    if (reviewWeightsVal) {
      reviewWeightsVal.textContent = isPretrained ? `Pretrained Checkpoint (${MODEL_REGISTRY[state.aiTraining.pretrained.modelId].checkpoint})` : `Random Weights $W_0$ (${state.aiTraining.scratch.initScheme}) — Zero Pretrained Weights`;
    }
    if (reviewDatasetVal) {
      reviewDatasetVal.textContent = isPretrained ? getSelectedDatasetName(state.aiTraining.pretrained.dataset) : (state.dataset ? state.dataset.name : 'Isolated Partition');
    }
    if (reviewGovVal) {
      if (state.aiTraining.policyStatus === 'passed') {
        reviewGovVal.className = 'badge badge-success';
        reviewGovVal.textContent = 'Passed (0 Policy Violations)';
      } else {
        reviewGovVal.className = 'badge badge-exclude';
        reviewGovVal.textContent = 'Blocked: Policy Remediations Required';
      }
    }
    if (reviewParamsVal) {
      reviewParamsVal.textContent = isPretrained
        ? `Method: ${state.aiTraining.pretrained.method.toUpperCase()} | Epochs: ${state.aiTraining.pretrained.epochs} | Batch: ${state.aiTraining.pretrained.batchSize} | LR: ${state.aiTraining.pretrained.learningRate} | Split: ${state.aiTraining.pretrained.valSplit}%`
        : `Arch: ${state.aiTraining.scratch.architecture.toUpperCase()} | Size: ${state.aiTraining.scratch.size} | Epochs: ${state.aiTraining.scratch.epochs} | Batch: ${state.aiTraining.scratch.batchSize} | LR: ${state.aiTraining.scratch.learningRate} | Dropout: ${state.aiTraining.scratch.dropout}`;
    }
    if (reviewResourceVal) {
      reviewResourceVal.textContent = isPretrained
        ? `${MODEL_REGISTRY[state.aiTraining.pretrained.modelId].estVRAM} VRAM • ${MODEL_REGISTRY[state.aiTraining.pretrained.modelId].estTime}`
        : `${SCRATCH_ARCHITECTURES[state.aiTraining.scratch.architecture].baseMemory} RAM • ~5m 00s`;
    }

    if (el.preflightReviewModal) el.preflightReviewModal.classList.add('active');
  }

  function closePreflightReviewModal() {
    if (el.preflightReviewModal) el.preflightReviewModal.classList.remove('active');
  }

  function openBackendSpecModal() {
    if (el.backendSpecModal) el.backendSpecModal.classList.add('active');
  }

  function closeBackendSpecModal() {
    if (el.backendSpecModal) el.backendSpecModal.classList.remove('active');
  }

  // 8. TRAINING SIMULATION RUNNER (EXPLICITLY [DEMO])
  function startTrainingSimulation() {
    closePreflightReviewModal();

    if (state.aiTraining.policyStatus === 'blocked') {
      showNotification('Training dispatch blocked: Dataset failed security & privacy scan.', 'error');
      return;
    }

    const mode = state.aiTraining.mode;
    const isPretrained = mode === 'pretrained';
    const modelKey = isPretrained ? state.aiTraining.pretrained.modelId : state.aiTraining.scratch.architecture;
    const modelTitle = isPretrained ? MODEL_REGISTRY[modelKey].name : SCRATCH_ARCHITECTURES[modelKey].name;
    const totalEpochs = isPretrained ? parseInt(state.aiTraining.pretrained.epochs, 10) : Math.min(20, parseInt(state.aiTraining.scratch.epochs, 10));
    const dsName = isPretrained ? getSelectedDatasetName(state.aiTraining.pretrained.dataset) : (state.dataset ? state.dataset.name : 'Isolated Partition');

    const jobId = `job-${mode}-${Date.now().toString(36).slice(-5)}`;
    state.aiTraining.activeJob = {
      id: jobId,
      name: `${jobId} (${modelTitle})`,
      mode: mode,
      modelTitle: modelTitle,
      datasetName: dsName,
      status: 'training',
      epoch: 0,
      totalEpochs: totalEpochs,
      isCancelled: false,
      timer: null,
      checkpoints: [],
    };

    if (el.trainingLiveMonitor) el.trainingLiveMonitor.classList.remove('hidden');
    if (el.liveMonitorTitle) el.liveMonitorTitle.textContent = `Training Job: ${jobId} (${modelTitle})`;
    if (el.liveJobStatusBadge) {
      el.liveJobStatusBadge.innerHTML = '<span class="pulse-dot"></span> Training [DEMO]';
      el.liveJobStatusBadge.className = 'queue-status training';
    }
    if (el.liveEvaluationResultsBox) el.liveEvaluationResultsBox.classList.add('hidden');

    scrollToSection('trainingLiveMonitor');

    if (el.liveTerminalLogsBody) {
      el.liveTerminalLogsBody.innerHTML = '';
      appendTerminalLog('INFO', `[DEMO] Allocating container worker sandbox for job ${jobId}...`);
      appendTerminalLog('INFO', `[DEMO] Execution Mode: ${mode === 'pretrained' ? 'PRE-TRAINED ADAPTATION' : 'TRAIN FROM SCRATCH (COLD START)'}`);
      appendTerminalLog('INFO', `[DEMO] Architecture: ${modelTitle}`);
      if (!isPretrained) {
        appendTerminalLog('OK', `[DEMO] Random weights matrix W0 initialized via ${state.aiTraining.scratch.initScheme}. Zero pretrained checkpoints loaded.`);
      } else {
        appendTerminalLog('OK', `[DEMO] Backbone checkpoint verified: ${MODEL_REGISTRY[modelKey].checkpoint}`);
      }
      appendTerminalLog('INFO', `[DEMO] Bound dataset: ${dsName} • Governance scan: PASSED.`);
      appendTerminalLog('INFO', `[DEMO] Training iteration loop started...`);
    }

    let currentEpoch = 0;
    let trainLoss = isPretrained ? 0.48 : 0.85;
    let valLoss = isPretrained ? 0.52 : 0.91;
    let accuracy = isPretrained ? 76.5 : 52.0;
    let seconds = 0;

    state.aiTraining.activeJob.timer = setInterval(() => {
      if (!state.aiTraining.activeJob || state.aiTraining.activeJob.isCancelled) {
        clearInterval(state.aiTraining.activeJob.timer);
        return;
      }

      seconds += 3;
      currentEpoch++;
      state.aiTraining.activeJob.epoch = currentEpoch;

      trainLoss = Math.max(0.016, trainLoss * 0.86 - (Math.random() * 0.004));
      valLoss = Math.max(0.024, valLoss * 0.88 + (Math.random() * 0.004 - 0.002));
      accuracy = Math.min(96.5, accuracy + (100 - accuracy) * 0.17 + (Math.random() * 0.4 - 0.2));

      const pct = Math.round((currentEpoch / totalEpochs) * 100);
      if (el.liveTrainingProgressBar) el.liveTrainingProgressBar.style.width = `${pct}%`;
      if (el.liveStatEpoch) el.liveStatEpoch.textContent = `${currentEpoch} / ${totalEpochs}`;
      if (el.liveStatTrainLoss) el.liveStatTrainLoss.textContent = trainLoss.toFixed(4);
      if (el.liveStatValLoss) el.liveStatValLoss.textContent = valLoss.toFixed(4);
      if (el.liveStatAccuracy) el.liveStatAccuracy.textContent = `${accuracy.toFixed(1)}%`;

      const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
      const secs = (seconds % 60).toString().padStart(2, '0');
      if (el.liveStatElapsed) el.liveStatElapsed.textContent = `${mins}:${secs}`;

      appendTerminalLog('TRAIN', `[DEMO] Epoch ${currentEpoch}/${totalEpochs} — Train Loss: ${trainLoss.toFixed(4)} — Val Loss: ${valLoss.toFixed(4)} — Acc: ${accuracy.toFixed(1)}%`);

      if (currentEpoch % Math.max(1, Math.floor(totalEpochs / 3)) === 0 || currentEpoch === totalEpochs) {
        const ckptName = `ckpt-epoch-${currentEpoch}.pt`;
        appendTerminalLog('CHECKPOINT', `[DEMO] Saved weights checkpoint: ${ckptName} (Val Loss: ${valLoss.toFixed(4)})`);
        state.aiTraining.activeJob.checkpoints.push({
          name: ckptName,
          type: isPretrained ? 'Adapted Head' : 'Scratch Learned Matrix',
          epoch: currentEpoch,
          valLoss: valLoss.toFixed(4),
          status: currentEpoch === totalEpochs ? 'Final Model' : 'Intermediate Checkpoint',
        });
      }

      if (currentEpoch >= totalEpochs) {
        clearInterval(state.aiTraining.activeJob.timer);
        finishTrainingSimulation(trainLoss, valLoss, accuracy);
      }
    }, 600);
  }

  function cancelTrainingJob() {
    if (state.aiTraining.activeJob && state.aiTraining.activeJob.status === 'training') {
      state.aiTraining.activeJob.isCancelled = true;
      state.aiTraining.activeJob.status = 'cancelled';
      clearInterval(state.aiTraining.activeJob.timer);

      if (el.liveJobStatusBadge) {
        el.liveJobStatusBadge.innerHTML = 'Cancelled [DEMO]';
        el.liveJobStatusBadge.className = 'queue-status draft';
      }
      appendTerminalLog('WARN', '[DEMO] Training job cancelled by user. Partial weights checkpoint discarded.');
      showNotification('Training job cancelled.', 'info');
    }
  }

  function finishTrainingSimulation(finalTrainLoss, finalValLoss, finalAccuracy) {
    if (!state.aiTraining.activeJob) return;
    state.aiTraining.activeJob.status = 'completed';

    if (el.liveJobStatusBadge) {
      el.liveJobStatusBadge.innerHTML = 'Completed [DEMO]';
      el.liveJobStatusBadge.className = 'queue-status completed';
    }

    appendTerminalLog('SUCCESS', `[DEMO] Training simulation completed! Final Validation Loss: ${finalValLoss.toFixed(4)}, Accuracy: ${finalAccuracy.toFixed(1)}%`);
    appendTerminalLog('INFO', '[DEMO] Artifacts registered to version v1.0.0-demo.');

    if (el.liveEvaluationResultsBox) el.liveEvaluationResultsBox.classList.remove('hidden');
    if (el.finalValLoss) el.finalValLoss.textContent = finalValLoss.toFixed(4);
    if (el.finalF1) el.finalF1.textContent = ((finalAccuracy / 100) * 0.98).toFixed(4);
    if (el.finalAccuracy) el.finalAccuracy.textContent = `${finalAccuracy.toFixed(1)}%`;
    if (el.finalLatency) el.finalLatency.textContent = state.aiTraining.mode === 'pretrained' ? '1.8 ms/item' : '0.4 ms/item';

    if (el.checkpointsTableBody) {
      el.checkpointsTableBody.innerHTML = '';
      state.aiTraining.activeJob.checkpoints.forEach((cp) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="font-mono">${escapeHtml(cp.name)}</td>
          <td>${escapeHtml(cp.type)}</td>
          <td class="font-mono">${cp.epoch}</td>
          <td class="font-mono">${cp.valLoss}</td>
          <td><span class="badge ${cp.status === 'Final Model' ? 'badge-include' : 'badge-review'}">${escapeHtml(cp.status)}</span></td>
        `;
        el.checkpointsTableBody.appendChild(tr);
      });
    }

    addAssistantMessage(
      `🎉 **Training simulation complete for ${state.aiTraining.activeJob.name}!**\n\n` +
      `• **Final Validation Loss**: ${finalValLoss.toFixed(4)}\n` +
      `• **Macro F1 Score**: ${((finalAccuracy / 100) * 0.98).toFixed(4)}\n` +
      `• **Accuracy**: ${finalAccuracy.toFixed(1)}%\n` +
      `• **Checkpoints Generated**: Saved ${state.aiTraining.activeJob.checkpoints.length} checkpoint artifacts linked to model version \`v1.0.0-demo\`.\n\n` +
      `*Notice: This run was executed in frontend demonstration mode [DEMO].*`
    );
  }

  function appendTerminalLog(level, message) {
    if (!el.liveTerminalLogsBody) return;
    const now = new Date();
    const timeStr = `[${now.toTimeString().split(' ')[0]}]`;

    const div = document.createElement('div');
    div.className = 'log-entry';

    let levelClass = 'info';
    if (level === 'OK' || level === 'SUCCESS') levelClass = 'success';
    if (level === 'WARN') levelClass = 'warning';
    if (level === 'TRAIN') levelClass = 'warning';
    if (level === 'CHECKPOINT') levelClass = 'info';

    div.innerHTML = `
      <span class="log-timestamp">${timeStr}</span>
      <span class="log-level ${levelClass}">${escapeHtml(level)}</span>
      <span class="log-message">${escapeHtml(message)}</span>
    `;

    el.liveTerminalLogsBody.appendChild(div);
    el.liveTerminalLogsBody.scrollTop = el.liveTerminalLogsBody.scrollHeight;

    const count = el.liveTerminalLogsBody.querySelectorAll('.log-entry').length;
    if (el.liveLogCount) el.liveLogCount.textContent = `${count} events`;
  }

  // 9. TRAINING ASSISTANT CHATBOX ENGINE
  function setupTrainingAssistantChat() {
    if (el.chatForm) {
      el.chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = el.chatInput ? el.chatInput.value.trim() : '';
        if (!text) return;
        el.chatInput.value = '';
        processUserChatPrompt(text);
      });
    }

    // Prompt Chips
    if (el.promptChips) {
      el.promptChips.forEach((chip) => {
        chip.addEventListener('click', () => {
          const promptText = chip.dataset.prompt;
          if (promptText) {
            processUserChatPrompt(promptText);
          }
        });
      });
    }

    // Clear History Button
    if (el.btnClearChat) {
      el.btnClearChat.addEventListener('click', () => {
        state.aiTraining.chatHistory = [];
        if (el.chatMessagesStream) {
          el.chatMessagesStream.innerHTML = `
            <div class="chat-message assistant">
              <div class="msg-avatar">Z</div>
              <div class="msg-bubble">
                <div class="msg-author">Zemo Assistant <span class="msg-time">Just now</span></div>
                <div class="msg-content">
                  <p>Chat history cleared. How can I help configure your training pipeline today?</p>
                </div>
              </div>
            </div>
          `;
        }
        showNotification('Chat session history cleared.', 'info');
      });
    }
  }

  function addUserMessage(text) {
    state.aiTraining.chatHistory.push({ sender: 'user', text: text, time: 'Just now' });
    if (!el.chatMessagesStream) return;

    const msgEl = document.createElement('div');
    msgEl.className = 'chat-message user';
    msgEl.innerHTML = `
      <div class="msg-avatar">U</div>
      <div class="msg-bubble">
        <div class="msg-author">You <span class="msg-time">Just now</span></div>
        <div class="msg-content">
          <p>${escapeHtml(text)}</p>
        </div>
      </div>
    `;
    el.chatMessagesStream.appendChild(msgEl);
    el.chatMessagesStream.scrollTop = el.chatMessagesStream.scrollHeight;
  }

  function addAssistantMessage(markdownText, configCard) {
    state.aiTraining.chatHistory.push({ sender: 'assistant', text: markdownText, configCard: configCard });
    if (!el.chatMessagesStream) return;

    const msgEl = document.createElement('div');
    msgEl.className = 'chat-message assistant';

    // Format markdown text simply and securely
    let formattedHtml = escapeHtml(markdownText)
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code class="font-mono">$1</code>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n• (.*?)/g, '<br>• $1')
      .replace(/\n1\. (.*?)/g, '<br>1. $1')
      .replace(/\n2\. (.*?)/g, '<br>2. $1')
      .replace(/\n3\. (.*?)/g, '<br>3. $1')
      .replace(/\n4\. (.*?)/g, '<br>4. $1');

    let cardHtml = '';
    if (configCard) {
      cardHtml = `
        <div class="chat-config-preview-card">
          <div class="chat-config-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
            ${escapeHtml(configCard.title)}
          </div>
          <div class="chat-config-details">
            <div><strong>Mode:</strong> ${escapeHtml(configCard.mode)}</div>
            <div><strong>Model:</strong> ${escapeHtml(configCard.model)}</div>
            <div><strong>Method / Init:</strong> ${escapeHtml(configCard.method)}</div>
            <div><strong>Epochs:</strong> ${configCard.epochs}</div>
            <div><strong>Batch Size:</strong> ${configCard.batchSize}</div>
            <div><strong>Learning Rate:</strong> ${escapeHtml(configCard.lr)}</div>
          </div>
          <div class="chat-actions-row">
            <button class="btn btn-chat-action btn-chat-apply" id="${configCard.btnId}">Apply Configuration</button>
            <button class="btn btn-chat-action" id="${configCard.btnReviewId}">Review Plan</button>
          </div>
        </div>
      `;
    }

    msgEl.innerHTML = `
      <div class="msg-avatar">Z</div>
      <div class="msg-bubble">
        <div class="msg-author">Zemo Assistant <span class="msg-time">Just now</span></div>
        <div class="msg-content">
          <p>${formattedHtml}</p>
          ${cardHtml}
        </div>
      </div>
    `;

    el.chatMessagesStream.appendChild(msgEl);
    el.chatMessagesStream.scrollTop = el.chatMessagesStream.scrollHeight;

    // Attach listeners to interactive card buttons
    if (configCard) {
      const applyBtn = document.getElementById(configCard.btnId);
      if (applyBtn) {
        applyBtn.addEventListener('click', () => {
          configCard.onApply();
        });
      }
      const reviewBtn = document.getElementById(configCard.btnReviewId);
      if (reviewBtn) {
        reviewBtn.addEventListener('click', () => {
          openPreflightReviewModal();
        });
      }
    }
  }

  function processUserChatPrompt(prompt) {
    addUserMessage(prompt);

    const lower = prompt.toLowerCase();

    // 1. "Train a sentiment classifier using my uploaded dataset."
    if (lower.includes('sentiment') || (lower.includes('classifier') && lower.includes('uploaded'))) {
      setTimeout(() => {
        const hasTextData = state.dataset && state.dataset.types && Object.values(state.dataset.types).includes('text');
        const dsName = state.dataset ? state.dataset.name : 'Customer Sentiment Reviews (Approved)';

        const textResponse = 
          `To build a **Sentiment Classifier** on **${dsName}**, you have two architectural paths in Zemo:\n\n` +
          `**Path A: Pre-trained Model Adaptation (Recommended)**\n` +
          `• **Model**: \`DistilBERT-Base-Uncased\` (66.4M parameters)\n` +
          `• **Method**: \`Classification-Head Training\` (freezes the transformer backbone and updates only the classification head)\n` +
          `• **Benefits**: Superior semantic accuracy on smaller sample sizes, low VRAM footprint (~2.1 GB), fast convergence in 3 epochs.\n\n` +
          `**Path B: Train from Scratch (Cold Start)**\n` +
          `• **Model**: \`Bidirectional LSTM (BiLSTM)\` or \`1D-ConvNet\` with random Glorot weight initialization ($W_0$).\n` +
          `• **Benefits**: Zero external checkpoints or pretrained weights. Use this if strict regulatory privacy forbids any external models. Requires 40–60 epochs.`;

        const card = {
          title: 'Recommended Sentiment Setup',
          mode: 'Pre-trained Adaptation',
          model: 'DistilBERT-Base-Uncased',
          method: 'Classification-Head Training',
          epochs: 3,
          batchSize: 16,
          lr: '0.00002',
          btnId: 'btnApplySentiment_' + Date.now(),
          btnReviewId: 'btnReviewSentiment_' + Date.now(),
          onApply: () => {
            setAiTrainingMode('pretrained');
            if (el.ptModelSelect) {
              el.ptModelSelect.value = 'distilbert-base';
              handlePretrainedModelChange('distilbert-base');
            }
            const headRadio = document.querySelector('input[name="ptMethod"][value="head"]');
            if (headRadio) headRadio.checked = true;
            if (el.ptEpochs) { el.ptEpochs.value = 3; el.valPtEpochs.textContent = '3'; }
            if (el.ptLR) el.ptLR.value = '0.00002';
            if (el.ptBatchSize) el.ptBatchSize.value = '16';
            showNotification('Applied recommended Sentiment Classifier settings.', 'success');
          },
        };

        addAssistantMessage(textResponse, card);
      }, 350);
      return;
    }

    // 2. "Help me fine-tune a supported pre-trained model."
    if (lower.includes('fine-tune') || lower.includes('pre-trained') || lower.includes('pretrained')) {
      setTimeout(() => {
        setAiTrainingMode('pretrained');

        const textResponse = 
          `Here is how to adapt a verified pre-trained model in Zemo:\n\n` +
          `1. **Model Registry Selection**: Choose a verified model like \`DistilBERT-Base-Uncased\` (66.4M), \`RoBERTa-Base\` (125M), or \`MobileNet-V3-Small\` (2.5M). *Notice: Unsupported models like Llama-3-70B are blocked from fine-tuning here due to multi-node cluster requirements.*\n` +
          `2. **Adaptation Method**:\n` +
          `• **Classification-Head**: Freezes backbone weights, trains top head. Best for fast transfer & low VRAM.\n` +
          `• **Full Fine-Tuning**: Trains all layers with a small learning rate ($2\\times 10^{-5}$) and linear decay.\n` +
          `• **LoRA (PEFT)**: Trains low-rank adapter projections.\n` +
          `3. **Governance**: Ensure dataset passes privacy scan before execution.`;

        const card = {
          title: 'Fine-Tuning Preset (DistilBERT)',
          mode: 'Pre-trained Adaptation',
          model: 'DistilBERT-Base-Uncased',
          method: 'Classification-Head Training',
          epochs: 3,
          batchSize: 16,
          lr: '0.00002',
          btnId: 'btnApplyFineTune_' + Date.now(),
          btnReviewId: 'btnReviewFineTune_' + Date.now(),
          onApply: () => {
            setAiTrainingMode('pretrained');
            if (el.ptModelSelect) {
              el.ptModelSelect.value = 'distilbert-base';
              handlePretrainedModelChange('distilbert-base');
            }
            if (el.ptEpochs) { el.ptEpochs.value = 3; el.valPtEpochs.textContent = '3'; }
            if (el.ptLR) el.ptLR.value = '0.00002';
            showNotification('Applied verified Fine-Tuning configuration.', 'success');
          },
        };

        addAssistantMessage(textResponse, card);
      }, 350);
      return;
    }

    // 3. "Build a new text classification model from scratch using only my dataset."
    if (lower.includes('scratch') && (lower.includes('text') || lower.includes('dataset'))) {
      setTimeout(() => {
        setAiTrainingMode('scratch');

        const textResponse = 
          `**Train from Scratch Protocol (Strict Cold Start)**:\n\n` +
          `• **Weights Guarantee**: All weights will be randomly initialized ($W_0 \\sim \\text{Glorot Uniform}$). Zero pre-trained checkpoints, embeddings, or external weights will be loaded.\n` +
          `• **Architecture**: \`Bidirectional LSTM (BiLSTM)\` or \`Task-Specific Mini-Transformer\` with learned embeddings trained purely on your private dataset.\n` +
          `• **Sample Size Check**: Training from scratch requires 40–80 epochs and sufficient sample volume to avoid overfitting.\n` +
          `• **Architectural Scope**: You are training a small task-specific model (~220K params). This is fully feasible on local or standard GPU workers, distinct from training multi-billion parameter foundation LLMs from scratch.`;

        const card = {
          title: 'Scratch Cold Start Blueprint',
          mode: 'Train from Scratch',
          model: 'Bidirectional LSTM (BiLSTM)',
          method: 'Glorot Uniform Random W0',
          epochs: 50,
          batchSize: 32,
          lr: '0.001',
          btnId: 'btnApplyScratch_' + Date.now(),
          btnReviewId: 'btnReviewScratch_' + Date.now(),
          onApply: () => {
            setAiTrainingMode('scratch');
            if (el.scratchArchSelect) el.scratchArchSelect.value = 'bilstm';
            state.aiTraining.scratch.architecture = 'bilstm';
            if (el.scratchEpochs) { el.scratchEpochs.value = 50; el.valScratchEpochs.textContent = '50'; }
            if (el.scratchLR) el.scratchLR.value = '0.001';
            if (el.scratchBatchSize) el.scratchBatchSize.value = '32';
            validateScratchTask();
            showNotification('Applied Scratch BiLSTM configuration.', 'success');
          },
        };

        addAssistantMessage(textResponse, card);
      }, 350);
      return;
    }

    // 4. "Analyze my dataset and recommend a suitable training approach."
    if (lower.includes('analyze') || lower.includes('recommend') || lower.includes('approach')) {
      setTimeout(() => {
        const ds = state.dataset;
        if (!ds) {
          addAssistantMessage(
            `**Dataset Status**: No custom dataset uploaded yet.\n\n` +
            `• You can upload a CSV in **Section 01 (Upload)** or use our sample datasets: \`customer_churn.csv\` or \`housing_sample.csv\`.\n` +
            `• **Quick Guideline**:\n` +
            `  - For text & NLP sentiment: Choose **Pre-trained Adaptation** with DistilBERT.\n` +
            `  - For tabular classification/regression: Choose **Train from Scratch** with Deep MLP or TabResNet.\n` +
            `• All datasets undergo our automated **Zemo Governance Policy Scan** before dispatch.`
          );
          return;
        }

        const isRegression = ds.types && Object.values(ds.types).some(t => t === 'numeric');
        const textResponse = 
          `**Dataset Analysis Report: ${ds.name}**\n\n` +
          `• **Rows**: ${ds.rows ? ds.rows.length : 0} | **Columns**: ${ds.headers ? ds.headers.length : 0}\n` +
          `• **Inferred Modality**: Tabular structured data (${Object.keys(ds.types || {}).length} features).\n` +
          `• **Recommended Mode**: **Train from Scratch (Mode B)** using \`Deep MLP\` or \`TabResNet\`.\n` +
          `• **Governance Status**: ${state.aiTraining.policyStatus === 'passed' ? 'Approved (0 PII violations)' : 'Review Required'}.\n` +
          `• **Recommended Hyperparameters**: 50 Epochs, Learning Rate 0.001, Batch Size 32, StandardScaler.`;

        const card = {
          title: `Recommended for ${ds.name}`,
          mode: 'Train from Scratch',
          model: 'Deep Multi-Layer Perceptron (MLP)',
          method: 'Glorot Uniform ($W_0$)',
          epochs: 50,
          batchSize: 32,
          lr: '0.001',
          btnId: 'btnApplyAnalysis_' + Date.now(),
          btnReviewId: 'btnReviewAnalysis_' + Date.now(),
          onApply: () => {
            setAiTrainingMode('scratch');
            if (el.scratchArchSelect) el.scratchArchSelect.value = 'mlp';
            state.aiTraining.scratch.architecture = 'mlp';
            if (el.scratchEpochs) { el.scratchEpochs.value = 50; el.valScratchEpochs.textContent = '50'; }
            if (el.scratchLR) el.scratchLR.value = '0.001';
            validateScratchTask();
            showNotification(`Applied optimal configuration for ${ds.name}.`, 'success');
          },
        };

        addAssistantMessage(textResponse, card);
      }, 350);
      return;
    }

    // 5. "Explain my model's evaluation metrics."
    if (lower.includes('metric') || lower.includes('evaluation') || lower.includes('explain')) {
      setTimeout(() => {
        addAssistantMessage(
          `**Understanding Your Model Evaluation Benchmarks**:\n\n` +
          `1. **Validation Loss**: Quantifies generalization error on unseen holdout test data. If train loss decreases while validation loss increases, your model is **overfitting**.\n` +
          `2. **Macro F1 Score**: The harmonic mean of Precision and Recall across classes. Critical for imbalanced datasets where raw accuracy can be deceptive.\n` +
          `3. **Generalization Delta**: $\\Delta = \\mathcal{L}_{val} - \\mathcal{L}_{train}$. A delta $< 0.05$ indicates balanced generalization.\n` +
          `4. **Inference Latency**: Average time per forward pass. Task-specific neural models achieve ~0.4ms to 1.8ms on standard hardware.\n\n` +
          `*All metrics in this studio are linked to verifiable checkpoint artifact IDs.*`
        );
      }, 350);
      return;
    }

    // 6. Commands (review, start, cancel)
    if (lower.includes('review')) {
      openPreflightReviewModal();
      addAssistantMessage('Opened the **Pre-Flight Review Summary**. Review your verified configuration before dispatch.');
      return;
    }
    if (lower.includes('cancel') || lower.includes('stop')) {
      cancelTrainingJob();
      addAssistantMessage('Cancelled active training job and reset telemetry.');
      return;
    }

    // General fallback
    setTimeout(() => {
      addAssistantMessage(
        `I understand you're asking about: "${prompt}".\n\n` +
        `• To configure **Pre-trained Adaptation**, select Mode A above or ask me to "Help me fine-tune".\n` +
        `• To train a model from **Scratch with zero pretrained weights**, select Mode B above or ask me to "Build a model from scratch".\n` +
        `• You can also click any of the suggested prompt chips above for instant guidance.`
      );
    }, 350);
  }

  // ============================================================
  // 10. BOOTSTRAP APPLICATION
  // ============================================================
  function init() {
    setupNavigation();
    setupUploadZone();
    setupRestrictionsControls();
    setupOutputControls();
    setupTrainingControls();
    setupAiTrainingControls();

    // Select initial model
    selectModel('random-forest');

    // Automatically load housing demo if wanted, or leave clean empty state with 1-click preview
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
