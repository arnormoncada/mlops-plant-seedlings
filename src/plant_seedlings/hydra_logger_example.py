import hydra
from omegaconf import DictConfig
from loguru import logger

# Import your custom logger setup
from plant_seedlings.logger import setup_logger


@hydra.main(version_base="1.2", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Example main function that uses Hydra and sets up Loguru.
    """
    # Retrieve Hydra's automatically generated output directory for this run
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Set up the logger using config values from config.yaml
    setup_logger(
        log_dir=hydra_output_dir,
        filename=cfg.logging.filename,
        rotation=cfg.logging.rotation,
        retention=cfg.logging.retention,
        level=cfg.logging.level,
        backtrace=cfg.logging.backtrace,  # <--- new
        colorize=cfg.logging.colorize,  # <--- new
    )

    # Now all logger messages go to the file + console sinks we configured above.
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning.")
    logger.error("This is an error.")
    logger.critical("This is critical.")

    # You can also log the entire config for quick reference
    logger.info(f"Hydra config contents:\n{cfg}")


if __name__ == "__main__":
    main()
