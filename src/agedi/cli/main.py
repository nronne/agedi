import rich_click as click
from agedi.cli.train import train
from agedi.cli.train_hydra import train_hydra
from agedi.cli.sample import sample
from agedi.cli.inspect import inspect

@click.group()
@click.version_option()
def cli() -> None:
    """Command Line Interface for the AGEDI package."""
    pass

cli.add_command(train)
cli.add_command(train_hydra)
cli.add_command(sample)
cli.add_command(inspect)

