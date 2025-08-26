import yaml
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

class PromptManager:
    def __init__(self, prompts_dir: str, config_path: str):
        # Initialize Jinja environment
        self.env = Environment(
            loader=FileSystemLoader(prompts_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        # Load the full nested prompts tree from config.yaml
        raw = yaml.safe_load(Path(config_path).read_text())
        self.config = raw.get('prompts', {})
        self.cache = {}

    def get(self, key: str, **overrides) -> str:
        """
        Fetch a prompt by a possibly-nested key, e.g. "CategoryName.subprompt",
        merging in any overrides on top of the template's defaults.
        """
        # Walk down the nested dict via the dot-separated key parts
        parts = key.split('.')
        node = self.config
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"Prompt key '{key}' not found in config.")
            node = node[part]

        # At this point, node should be a dict with 'template' and optional 'defaults'
        try:
            template_name = node['template']
        except (TypeError, KeyError):
            raise KeyError(f"Prompt key '{key}' does not refer to a prompt entry.")
        defaults = node.get('defaults', {})

        # Cache and load the Jinja template
        template = self.cache.get(key)
        if template is None:
            template = self.env.get_template(template_name)
            self.cache[key] = template

        # Merge defaults + overrides and render
        context = {**defaults, **overrides}
        return template.render(**context)


if __name__ == '__main__':
    # Adjust these paths to point at your prompts directory and config file
    base_dir = Path(r"C:\Users\sayee\UTN_Projects\Multi3Hate\inference\prompts")
    cfg_file = base_dir / 'config.yaml'

    pm = PromptManager(str(base_dir), str(cfg_file))

    # Examples of fetching nested prompts:
    print("=== CoreCulturalUnderstanding.general ===")
    print(pm.get('CoreCulturalUnderstanding.general', country='India'))
    print("\n=== SensitivityModeration.tabooDetection ===")
    print(pm.get('SensitivityModeration.tabooDetection', country='India'))
    print("\n=== RegionalLinguisticAdaptation.iconography ===")
    print(pm.get('RegionalLinguisticAdaptation.iconography'))