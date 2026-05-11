import numpy as np


def ensemble_predict(cnn_model, vgg19_model, test_generator):
    """Generate ensemble predictions using average probability voting.

        Args:
                cnn_model: Trained CNN+CBAM Keras model.
                        vgg19_model: Trained VGG19 Keras model.
                                test_generator: Keras ImageDataGenerator test generator.

                                    Returns:
                                            Tuple of (ensemble_probs, final_predictions, true_labels)
                                                """
                                                    cnn_probs = cnn_model.predict(test_generator).flatten()
                                                        vgg19_probs = vgg19_model.predict(test_generator).flatten()

                                                            # Average probability voting
                                                                ensemble_probs = (cnn_probs + vgg19_probs) / 2
                                                                    final_predictions = (ensemble_probs > 0.5).astype(int)
                                                                        true_labels = test_generator.classes

                                                                            accuracy = np.mean(final_predictions == true_labels)
                                                                                print(f"Ensemble Accuracy: {accuracy * 100:.2f}%")

                                                                                    return ensemble_probs, final_predictions, true_labels
                                                                                    
