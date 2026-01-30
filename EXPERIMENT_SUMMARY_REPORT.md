# Decision Tree Experiments - Comprehensive Summary Report

Generated: 2026-01-30 20:55:29

================================================================================

## 1. Main Decision Tree Experiments

### Dataset Information

#### SensorNetGuard IDS

- **Training samples**: 8,000
- **Test samples**: 2,000
- **Features**: 17
- **Training class distribution**: {'0': 7610, '1': 390}
- **Test class distribution**: {'0': 1903, '1': 97}

#### Farm-Flow Binary Classification

- **Training samples**: 561,081
- **Test samples**: 3,545
- **Features**: 30
- **Training class distribution**: {'0': 284880, '1': 276201}
- **Test class distribution**: {'0': 1774, '1': 1771}

#### CIC IDS 2017 Binary Classification

- **Training samples**: 2,264,594
- **Test samples**: 566,149
- **Features**: 78
- **Training class distribution**: {'0': 1818477, '1': 446117}
- **Test class distribution**: {'0': 454620, '1': 111529}

### Tree Properties

| Dataset | Max Depth | Nodes | Leaves |
|---------|-----------|-------|--------|
| SensorNetGuard | 7 | 23 | 12 |
| Farm-Flow | 9 | 41 | 21 |
| CIC IDS 2017 | 43 | 6,369 | 3,185 |

### Test Set Performance Metrics

| Dataset | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Avg Precision |
|---------|----------|-----------|--------|----------|---------|---------------|
| SensorNetGuard | 0.9995 | 0.9898 | 1.0000 | 0.9949 | 0.9997 | 0.9898 |
| Farm-Flow | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CIC IDS 2017 | 0.9988 | 0.9969 | 0.9970 | 0.9969 | 0.9985 | 0.9951 |

### Decision Tree Visualizations

## 2. Decision Stump Experiments (1-Rule Models)

These experiments test model robustness by using only the top feature.


## 3. CIC IDS 2017 Depth-Limited Experiment

Comparison of unlimited depth vs max_depth=10:


| Metric | Unlimited Depth | max_depth=10 | Difference |
|--------|-----------------|--------------|------------|
| Accuracy | 0.9988 | 0.9968 | +0.0019 |
| Precision | 0.9969 | 0.9950 | +0.0019 |
| Recall | 0.9970 | 0.9890 | +0.0080 |
| F1_score | 0.9969 | 0.9920 | +0.0050 |
| Roc_auc | 0.9985 | 0.9958 | +0.0027 |

## 4. Feature Ablation Experiments

These experiments show how performance changes as features are removed, starting from lowest importance.


## 5. Top 10 Most Important Features (by Dataset)

### SensorNetGuard

1. **Error_Rate**: 0.842579
2. **Energy_Consumption_Rate**: 0.099186
3. **Data_Throughput**: 0.028865
4. **Packet_Drop_Rate**: 0.013451
5. **Data_Transmission_Frequency**: 0.008375
6. **Route_Request_Frequency**: 0.005748
7. **CPU_Usage**: 0.001797
8. **Packet_Rate**: 0.000000
9. **Memory_Usage**: 0.000000
10. **Data_Reception_Frequency**: 0.000000

### Farm-Flow

1. **orig_pkts**: 0.854616
2. **traffic**: 0.135946
3. **bwd_pkts_payload.tot**: 0.003808
4. **orig_ip_bytes**: 0.002836
5. **bwd_pkts_payload.avg**: 0.001091
6. **fwd_pkts_per_sec**: 0.000868
7. **data_pkts_difference**: 0.000737
8. **resp_pkts**: 0.000081
9. **fwd_iat.avg**: 0.000017
10. **pkts_difference**: 0.000000

### CIC IDS 2017

1. **Bwd Packet Length Std**: 0.376008
2. **Average Packet Size**: 0.189682
3. **Bwd Header Length**: 0.132329
4. **Destination Port**: 0.105178
5. **Max Packet Length**: 0.092449
6. **Init_Win_bytes_forward**: 0.027093
7. **min_seg_size_forward**: 0.018743
8. **Fwd Packet Length Std**: 0.009505
9. **Init_Win_bytes_backward**: 0.007099
10. **Fwd IAT Min**: 0.005158

## 6. Confusion Matrices (Test Set)

### SensorNetGuard

```
                Predicted
              Benign  Attack
Actual Benign    1902       1
       Attack       0      97
```

### Farm-Flow

```
                Predicted
              Benign  Attack
Actual Benign    1774       0
       Attack       0    1771
```

### CIC IDS 2017

```
                Predicted
              Benign  Attack
Actual Benign   454271     349
       Attack     336   111193
```
