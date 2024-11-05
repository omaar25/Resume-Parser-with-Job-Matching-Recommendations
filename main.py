from src.project.components.data_processing import DataProcessing
from src.project.components.model_training import ModelTraining
from src.project.components.model_evaluation import ModelEvaluation


if __name__ == "__main__":
    
    data_processor = DataProcessing()
    data_processor.process()

    model_trainer = ModelTraining()
    model_trainer.train()

    evaluation = ModelEvaluation()
    evaluation.evaluate()