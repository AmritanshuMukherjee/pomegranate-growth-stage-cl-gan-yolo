import yaml, logging
from models.yolo.yolo_cl import YOLOCurriculum
from utils.train_utils import train_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def main():
    logging.info("Training YOLO + Curriculum Learning")

    dataset = yaml.safe_load(open("config/dataset.yaml"))["dataset"]
    cfg = yaml.safe_load(open("config/train_common.yaml"))["training"]

    model = YOLOCurriculum(
        num_classes=dataset["num_classes"],
        input_size=dataset["image_size"],
        model_size="medium"
    )

    train_loop(
        model=model,
        dataset_cfg=dataset,
        train_cfg=cfg,
        experiment_name="yolo_curriculum",
        use_curriculum=True
    )

if __name__ == "__main__":
    main()
