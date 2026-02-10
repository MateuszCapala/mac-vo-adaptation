python3 MACVO.py --useRR --odom Config/Experiment/MACVO/MACVO_Performant.yaml --data Config/Sequence/RoverTraverse_Rectified.yaml

# Porównanie z GT - png
PYTHONPATH=/home/macvo/workspace python3 Evaluation/PlotSeq.py --spaces Results/MACVO-Performant@RoverTraverse_Rectified/02_10_141339

# Porównanie z GT - wizualizacja
python3 MACVO.py --useRR --odom Config/Experiment/MACVO/MACVO_Fast.yaml --data Config/Sequence/RoverTraverse_Rectified.yaml