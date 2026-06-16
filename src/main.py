from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from model_utils import add_stacking_model, build_model_registry, cross_validate_models, fit_pipeline
from preprocessing import build_preprocessor, load_dataset, split_features_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Treina e avalia modelos de regressao para previsao do desgaste "
            "de ferramentas a partir de variaveis experimentais."
        )
    )
    parser.add_argument("--data", required=True, help="Caminho para o arquivo CSV experimental.")
    parser.add_argument("--target", default="Desgaste", help="Nome da coluna alvo.")
    parser.add_argument(
        "--features",
        default=None,
        help="Lista de variaveis de entrada separadas por virgula. Se omitida, usa colunas numericas.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Numero de folds da validacao cruzada.")
    parser.add_argument("--random-state", type=int, default=42, help="Semente de reprodutibilidade.")
    parser.add_argument(
        "--output-dir",
        default="reports/experiment",
        help="Diretorio para salvar metricas, predicoes e graficos.",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Ignora modelos opcionais como XGBoost e LightGBM.",
    )
    parser.add_argument(
        "--no-stacking",
        action="store_true",
        help="Nao inclui o ensemble por stacking na avaliacao.",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Gera grafico SHAP para o melhor modelo quando a biblioteca estiver disponivel.",
    )
    return parser


def save_prediction_plot(predictions: pd.DataFrame, best_model: str, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib nao instalado. Grafico predicted_vs_real.png nao foi gerado.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(predictions["real"], predictions[best_model], alpha=0.85)

    minimum = min(predictions["real"].min(), predictions[best_model].min())
    maximum = max(predictions["real"].max(), predictions[best_model].max())
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="black", linewidth=1)

    plt.xlabel("Desgaste real")
    plt.ylabel("Desgaste previsto")
    plt.title(f"Valores reais vs. previstos - {best_model}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_vs_real.png", dpi=180)
    plt.close()


def save_shap_summary(pipeline, X: pd.DataFrame, feature_names: list[str], output_dir: Path) -> None:
    try:
        import shap
    except ImportError:
        print("SHAP nao instalado. Instale a dependencia ou execute sem --shap.")
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed_X = preprocessor.transform(X)

    explainer = shap.Explainer(model.predict, transformed_X, feature_names=feature_names)
    shap_values = explainer(transformed_X)

    shap.summary_plot(shap_values, transformed_X, feature_names=feature_names, show=False)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data)
    X, y, feature_names = split_features_target(data, target=args.target, features=args.features)
    preprocessor = build_preprocessor(feature_names)

    models = build_model_registry(
        random_state=args.random_state,
        include_optional=not args.skip_optional,
    )
    if not args.no_stacking:
        models = add_stacking_model(models, random_state=args.random_state)

    metrics, predictions = cross_validate_models(
        X=X,
        y=y,
        preprocessor=preprocessor,
        models=models,
        n_splits=args.folds,
        random_state=args.random_state,
    )

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    best_model_name = metrics.iloc[0]["model"]
    save_prediction_plot(predictions, best_model_name, output_dir)

    print(metrics.to_string(index=False))
    print(f"\nMelhor modelo: {best_model_name}")
    print(f"Arquivos gerados em: {output_dir}")

    if args.shap:
        best_pipeline = fit_pipeline(X, y, preprocessor, models[best_model_name])
        save_shap_summary(best_pipeline, X, feature_names, output_dir)


if __name__ == "__main__":
    main()
