import os.path

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib


COLOR_NAME_TO_RGB = {
    "red": "#cc202b",
    "green": "#2d982f",
    "blue": "#276aa8",
    "orange": "#fa7427"
}

ALL_TOKENS_TO_BE_VISUALIZED = ["answer_1", "answer_2", "answer_3", "subject"]

TEMPLATE_NUM = 3


def get_model_name_for_visualization(model_name):
    if "Llama" in model_name:
        return "Llama-3-8B-Instruct"
    else:
        return "Mistral-7B-Instruct"


def reformat_var_name_for_visualization(var_name):
    var_name = var_name.replace("_", " ")
    var_name = " ".join([word.capitalize() for word in var_name.split(" ")])
    return var_name


def get_component_full_name(component_name):
    if "attn" in component_name:
        return "Attention"
    elif "mlp" in component_name:
        return "MLP"
    elif "hidden_state" in component_name:
        return "Hidden State"
    else:
        raise ValueError(f"Unknown component_name: {component_name}")


def merge_figures(subfigure_fn_list, title, output_fn, exp_name):
    figure_num = len(subfigure_fn_list)
    image_obj_list = [Image.open(fn) for fn in subfigure_fn_list]
    image_size_list = [image_obj.size for image_obj in image_obj_list]
    fig_size = (sum([size[0] for size in image_size_list])/100, max([size[1] for size in image_size_list])/100)
    assert figure_num > 1
    fig, axes = plt.subplots(1, figure_num, figsize=fig_size)
    for i, ax in enumerate(axes):
        ax.imshow(image_obj_list[i])
        ax.axis('off')

    if "independence" not in exp_name:
        plt.plot([], [], marker='*', label='Subject', color=COLOR_NAME_TO_RGB['red'])
        plt.plot([], [], marker='o', label='Answer 1', color=COLOR_NAME_TO_RGB['blue'])
        plt.plot([], [], marker='s', label='Answer 2', color=COLOR_NAME_TO_RGB['orange'])
        plt.plot([], [], marker='^', label='Answer 3', color=COLOR_NAME_TO_RGB['green'])

    if exp_name == "decode_attn_mlp":
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.6, 1.04),
            ncol=4,
            fontsize=14,
        )
        fig.suptitle(title, fontsize=24, y=0.97)
        fig.tight_layout(pad=0)
        fig.subplots_adjust(top=0.95)
    elif exp_name == "token_lens_attn_knockout":
        plt.legend(
            loc='upper right',
            bbox_to_anchor=(1, 1.06),
            ncol=4,
            fontsize=14,
        )
        fig.suptitle(title, fontsize=24, y=0.96)
        fig.subplots_adjust(top=0.80)
        fig.tight_layout(pad=0)
    elif exp_name == "head_promotion_suppression_rate":
        fig.suptitle(title, fontsize=24, y=0.97)
        fig.tight_layout(pad=0)
        fig.subplots_adjust(top=0.95)
    else:
        raise NotImplementedError(f"Unknown exp_name for merging images: {exp_name}")

    output_dir = os.path.dirname(output_fn)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_fn)
    matplotlib.pyplot.close()


def merge_dataset_model_specific_token_level_figures(subfigure_fn_list, main_title, title1, title2, output_fn):
    """
    Merge all six figures together and put them in one row: three on the left, three on the right.
    Add title1 above the left three figures, title2 above the right three figures.
    Place the legend on the right and add a dashed line between the two groups of figures.
    """
    assert len(subfigure_fn_list) == 6, "Exactly six subfigures are required."

    image_obj_list = [Image.open(fn) for fn in subfigure_fn_list]
    image_width, image_height = image_obj_list[0].size
    image_size_list = [image_obj.size for image_obj in image_obj_list]
    fig_size = (sum([size[0] for size in image_size_list]) / 100, max([size[1] for size in image_size_list]) / 100)
    plt.fontsize = 26
    fig, axes = plt.subplots(1, 6, figsize=fig_size)

    for i, ax in enumerate(axes):
        ax.imshow(image_obj_list[i])
        ax.axis('off')
    fig.tight_layout(pad=0)

    # Add dashed line between the two groups of figures
    mid_x = (3 * image_width) / (6 * image_width)
    fig.add_artist(plt.Line2D([mid_x, mid_x], [0.0, 0.95], color='grey', linestyle='--', transform=fig.transFigure))

    plt.text(0.27, 0.95, title1, fontsize=30, ha='center', va='center', transform=fig.transFigure)
    plt.text(0.75, 0.95, title2, fontsize=30, ha='center', va='center', transform=fig.transFigure)
    fig.suptitle(main_title, fontsize=30, y=1.1)
    fig.subplots_adjust(top=0.9)

    # Add legend
    plt.plot([], [], marker='*', label='Subject', color=COLOR_NAME_TO_RGB['red'])
    plt.plot([], [], marker='o', label='Answer 1', color=COLOR_NAME_TO_RGB['blue'])
    plt.plot([], [], marker='s', label='Answer 2', color=COLOR_NAME_TO_RGB['orange'])
    plt.plot([], [], marker='^', label='Answer 3', color=COLOR_NAME_TO_RGB['green'])
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(0.95, 1.15),
        ncol=2,
        fontsize=18,
    )

    output_dir = os.path.dirname(output_fn)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_fn, bbox_inches='tight')
    plt.close()
    matplotlib.pyplot.close()
