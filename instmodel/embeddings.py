import pandas as pd
import numpy as np

from .instruction_model import create_instructions_model_from_transformation_list
from .model import (
    InputBuffer,
    Dense,
    Concatenate,
    ModelGraph,
)


def one_hot_encode(data, one_hot, k):
    one_hot_columns = one_hot.columns
    val_one_hot = pd.get_dummies(data[k], prefix=k, dtype=np.float32).reindex(
        columns=one_hot_columns, fill_value=0
    )

    assert val_one_hot.shape[1] == one_hot.shape[1]
    assert (val_one_hot.columns == one_hot.columns).all()
    if not set(val_one_hot.columns).issubset(set(one_hot_columns)):
        raise ValueError(
            f"Validation set contains IDs not present in training set: {set(val_one_hot.columns) - set(one_hot_columns)}"
        )

    data = data.join(val_one_hot)
    data = data.drop(k, axis=1)

    return data


def embbed_predict(data, one_hot, model):
    one_hot_columns = one_hot.columns
    idxs = data[one_hot_columns].idxmax(axis=1)

    return embbed_predict_idx(idxs, one_hot.shape[1], model)


def embbed_predict_idx(idxs, one_hot_len, model):
    if isinstance(idxs, int):
        one_hot = np.zeros((1, one_hot_len), dtype=np.float32)
        one_hot[0, idxs] = 1.0
        return model.predict_on_batch(one_hot)[0]

    one_hot = np.zeros((len(idxs), one_hot_len), dtype=np.float32)
    one_hot[np.arange(len(idxs)), idxs] = 1.0

    return model.predict_on_batch(one_hot)


def get_embeddings_dict(emb_models):
    embeddings_dict = {}

    map_key_sorted = sorted(emb_models.keys())
    for k in map_key_sorted:
        sub_model = emb_models[k]
        model_info = sub_model["model"]
        one_hot_df = sub_model["one_hot"]

        category_embeddings = {}

        for idx, col in enumerate(one_hot_df.columns):
            prefix = k + "_"
            if not col.startswith(prefix):
                raise ValueError(
                    f"Column '{col}' does not start with expected prefix '{prefix}'"
                )

            cat_id_str = col[len(prefix) :]  # Remove prefix
            cat_id = round(float(cat_id_str))

            embedding_vector = embbed_predict_idx(
                idx, one_hot_df.shape[1], model_info
            ).tolist()
            category_embeddings[cat_id] = embedding_vector

        embeddings_dict[k] = category_embeddings
    return embeddings_dict


class EmbeddingWrapper:
    def __init__(
        self,
        training_data,
        validation_data,
        input_features,
        output_features,
        mapping,
        fraction=1.0,
    ):
        assert isinstance(training_data, pd.DataFrame) and isinstance(
            validation_data, pd.DataFrame
        )
        self.map_key_sorted = sorted(mapping.keys())
        self.original_training_data = training_data.copy()
        self.original_validation_data = validation_data.copy()
        self.mapping = mapping
        self.initial_input_features = input_features
        self.initial_output_features = output_features
        self.fraction = fraction

        self.create_one_hot_encoding()
        self.create_emb_step()

    def create_instruction_model(self):
        embeddings_dict = get_embeddings_dict(self.emb_models)

        transformation_list = []

        # The index where one-hot encoded columns start, based on how you constructed 'features'
        data_index = len(self.initial_input_features) - len(self.mapping)
        segment_start = data_index

        current_columns = list(self.s1_training_data_X.columns)
        self.s2_training_data_X = self.s1_training_data_X.copy()
        self.s2_validation_data_X = self.s1_validation_data_X.copy()

        for k, v in self.mapping.items():
            # Validate segment_start and segment_size
            if k not in self.emb_models or "one_hot" not in self.emb_models[k]:
                raise ValueError(f"No one_hot info found for key {k}")

            segment_size = self.emb_models[k]["one_hot"].shape[1]

            if segment_start + segment_size > len(current_columns):
                raise IndexError(
                    f"Segment for key '{k}' exceeds the number of columns available.\n"
                    f"segment_start: {segment_start}, segment_size: {segment_size}, total_cols: {len(current_columns)}"
                )

            one_hot_cols = current_columns[segment_start : segment_start + segment_size]

            # Sanity check that these columns exist in the DataFrame
            for col in one_hot_cols:
                if (
                    col not in self.s1_training_data_X.columns
                    or col not in self.s1_validation_data_X.columns
                ):
                    raise KeyError(f"Column {col} expected but not found in data.")

            # Extract the segments
            train_segment = self.s1_training_data_X[one_hot_cols]
            valid_segment = self.s1_validation_data_X[one_hot_cols]

            # Predict the embeddings using embeddings_dict
            train_embedded = self.emb_models[k]["model"].predict(train_segment)
            valid_embedded = self.emb_models[k]["model"].predict(valid_segment)

            print(f"Train embedded shape for {k}: {train_embedded.shape}")
            print(f"Validation embedded shape for {k}: {valid_embedded.shape}")

            # Check embedding shapes
            if train_embedded.shape[1] != v:
                raise ValueError(
                    f"Expected embeddings of size {v}, but got {train_embedded.shape[1]} for train data."
                )
            if valid_embedded.shape[1] != v:
                raise ValueError(
                    f"Expected embeddings of size {v}, but got {valid_embedded.shape[1]} for validation data."
                )

            # Create new column names for the embedded features
            embedded_col_names = [f"{k}_{i}" for i in range(v)]

            # Create DataFrames for the embedded features
            train_embedded_df = pd.DataFrame(
                train_embedded,
                index=self.s2_training_data_X.index,
                columns=embedded_col_names,
            )
            valid_embedded_df = pd.DataFrame(
                valid_embedded,
                index=self.s2_validation_data_X.index,
                columns=embedded_col_names,
            )

            # Rebuild the DataFrames with the embedded columns replacing the one-hot columns
            train_before = self.s2_training_data_X.iloc[:, :segment_start]
            train_after = self.s2_training_data_X.iloc[
                :, segment_start + segment_size :
            ]
            valid_before = self.s2_validation_data_X.iloc[:, :segment_start]
            valid_after = self.s2_validation_data_X.iloc[
                :, segment_start + segment_size :
            ]

            # Construct new DataFrames rather than modifying in place
            new_data_train_X = pd.concat(
                [train_before, train_embedded_df, train_after], axis=1
            )
            new_validation_data_X = pd.concat(
                [valid_before, valid_embedded_df, valid_after], axis=1
            )

            # Ensure consistent dtypes
            new_data_train_X = new_data_train_X.astype(np.float32, copy=False)
            new_validation_data_X = new_validation_data_X.astype(np.float32, copy=False)

            # Update the main DataFrames and the list of columns
            self.s2_training_data_X = new_data_train_X
            self.s2_validation_data_X = new_validation_data_X
            current_columns = list(self.s2_training_data_X.columns)

            default = embeddings_dict[k].pop(-1)

            # Append transformations
            transformation_list.append(
                {
                    "from": k,
                    "to": embedded_col_names,
                    "map": embeddings_dict[k],
                    "size": v,
                    "default": default,
                }
            )

            transformation_list.append({"delete": k})

            # Update segment_start to jump over the newly embedded columns
            # Originally we had one-hot columns replaced by 'v' embeddings:
            # So we move forward by `v - segment_size` because we've effectively
            # replaced `segment_size` columns with `v` columns, and these columns
            # start at the same segment_start position.
            segment_start = segment_start + v

        print("Final Data Train Columns:", self.s2_training_data_X.columns)
        print("Final Validation Data Columns:", self.s2_validation_data_X.columns)

        transformation_model, new_features = (
            create_instructions_model_from_transformation_list(
                self.initial_input_features, transformation_list
            )
        )

        self.s2_input_features = new_features
        self.s2_output_features = self.initial_output_features
        self.s2_training_data_y = self.s1_training_data_y
        self.s2_validation_data_y = self.s1_validation_data_y
        return transformation_model

    def create_one_hot_encoding(self):
        self.emb_models = {}
        maps = self.mapping
        training_data_X = self.original_training_data[
            self.initial_input_features
        ].copy()
        training_data_y = self.original_training_data[
            self.initial_output_features
        ].copy()
        validation_data_X = self.original_validation_data[
            self.initial_input_features
        ].copy()

        new_data_x = []
        new_data_y = []
        for k in self.map_key_sorted:
            subset_X = training_data_X.sample(frac=self.fraction).copy()
            subset_y = training_data_y.loc[subset_X.index].copy()

            # Modify the subset
            subset_X[k] = -1.0

            # Append the subset to the lists
            new_data_x.append(subset_X)
            new_data_y.append(subset_y)

        # Extend data_train_X and data_train_y with the new data
        training_data_X = pd.concat(
            [training_data_X] + new_data_x, ignore_index=True
        )  # Ensure consistent indexing
        training_data_y = pd.concat(
            [training_data_y] + new_data_y, ignore_index=True
        )  # Ensure consistent indexing

        for k in self.map_key_sorted:
            v = maps[k]

            one_hot = pd.get_dummies(training_data_X[k], prefix=k, dtype=np.float32)
            one_hot_columns = one_hot.columns

            assert f"{k}_-1.0" in one_hot_columns

            training_data_ids = set(training_data_X[k].unique())
            validation_data_ids = set(validation_data_X[k].unique())

            unique_validation_data_ids = validation_data_ids - training_data_ids

            # replace the unique_validation_data_ids (unknown) with -1
            validation_data_X.loc[
                validation_data_X[k].isin(unique_validation_data_ids), k
            ] = -1.0

            training_data_ids = set(training_data_X[k].unique())
            validation_data_ids = set(validation_data_X[k].unique())

            unique_validation_data_ids = (
                set(validation_data_X[k].unique()) - training_data_ids
            )

            print(
                f"Traning data x had {len(training_data_X.columns)} cols:\n{training_data_X.columns}"
            )
            print(f"Adding one hot {len(one_hot.columns)} columns and dropping {k}")

            training_data_X = training_data_X.join(one_hot)
            training_data_X = training_data_X.drop(k, axis=1)

            print(
                f"Traning data x now has {len(training_data_X.columns)} cols:\n{training_data_X.columns}"
            )

            validation_data_X = one_hot_encode(validation_data_X, one_hot, k)

            # Create model for embedding the one-hot columns
            inputs = InputBuffer(one_hot.shape[1])
            outputs = Dense(v, activation="sigmoid")(inputs)
            model = ModelGraph(inputs, outputs)

            self.emb_models[k] = {
                "model": model,
                "one_hot": one_hot,
            }

            assert "complete" not in training_data_X.columns

        self.s1_input_features = training_data_X.columns
        self.s1_output_features = training_data_y.columns
        self.s1_training_data_X = training_data_X
        self.s1_training_data_y = training_data_y
        self.s1_validation_data_X = validation_data_X
        self.s1_validation_data_y = self.original_validation_data[
            self.initial_output_features
        ].copy()

    def create_emb_step(self):
        full_inputs = InputBuffer(self.s1_training_data_X.shape[1])
        data_index = len(self.initial_input_features) - len(self.mapping)

        preprocessed_outputs = [full_inputs[:data_index]]

        for k in self.map_key_sorted:
            segment_size = self.emb_models[k]["one_hot"].shape[1]
            preprocessed_outputs.append(
                self.emb_models[k]["model"](
                    full_inputs[data_index : data_index + segment_size]
                )
            )
            data_index += segment_size

        assert data_index == self.s1_training_data_X.shape[1]

        full_preprocessed_outputs = Concatenate()(preprocessed_outputs)

        self.apply_smart_embeddings = ModelGraph(full_inputs, full_preprocessed_outputs)

        self.apply_embeddings = self.apply_smart_embeddings.get_keras()

        self.step1_size = data_index
        self.step2_size = full_preprocessed_outputs.os
