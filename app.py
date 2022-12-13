from flask import Flask, render_template, request
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    with open("xgb.pkl", mode="rb") as fp:
        model = pickle.load(fp)
    f_names = model.feature_names
    sex = "性別を選択してください"
    father = "1代祖を選択してください"
    grand = "2代祖を選択してください"
    great = "3代祖を選択してください"
    got = "4代祖を選択してください"
    return render_template(
        "index.html",
        f_names=f_names,
        sex=sex,
        father=father,
        grand=grand,
        great=great,
        got=got,
    )


@app.route("/cow_much", methods=["POST"])
def cow_much():
    with open("xgb.pkl", mode="rb") as fp:
        model = pickle.load(fp)

    f_names = model.feature_names
    test_df = pd.DataFrame(index=[1], columns=f_names)
    test_df = test_df.fillna(0)

    sex = request.form.get("sex")
    father = request.form.get("father")
    grand = request.form.get("grand")
    great = request.form.get("great")
    got = request.form.get("got")

    test_df[f"性別_{sex}"] = 1
    test_df[f"父牛_{father}"] = 1
    test_df[f"母の父_{grand}"] = 1
    test_df[f"母の祖父_{great}"] = 1
    test_df[f"母の祖祖父_{got}"] = 1

    def price_predict(df):
        with open("xgb.pkl", mode="rb") as fp:
            model_1 = pickle.load(fp)

        with open("random.pkl", mode="rb") as fp:
            model_2 = pickle.load(fp)

        with open("meta.pkl", mode="rb") as fp:
            meta_model = pickle.load(fp)

        df_tail = df.tail(1)
        model_1_pred = model_1.predict(xgb.DMatrix(df_tail))
        model_2_pred = model_2.predict(df_tail)
        stack_pred = np.column_stack((model_1_pred, model_2_pred))
        meta_pred = meta_model.predict(stack_pred)
        return int(meta_pred)

    price = price_predict(test_df)
    result = f"{int(price)}円"
    return render_template(
        "index.html",
        result=result,
        f_names=f_names,
        sex=sex,
        father=father,
        grand=grand,
        great=great,
        got=got,
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
