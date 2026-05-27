def print_cnf_mtx(mtx):
    print("\t\t       Predicted")
    print("\t\t         N   P")
    print(f"\t\tTrue  N {mtx[0, 0]}  {mtx[0, 1]}")
    print(f"\t\t      P {mtx[1, 0]}  {mtx[1, 1]}")


def print_cnf_mtx_CT(mtx):
    print("\t\t                Predicted")
    print("\t\t         1   2   3   4   5   6   7")
    print(
        f"\t\t      1 {mtx[0, 0]}  {mtx[0, 1]}  {mtx[0, 2]}   {mtx[0, 3]}   {mtx[0, 4]}  {mtx[0, 5]}   {mtx[0, 6]}"
    )
    print(
        f"\t\t      2 {mtx[1, 0]}  {mtx[1, 1]}  {mtx[1, 2]}  {mtx[1, 3]}   {mtx[1, 4]}  {mtx[1, 5]}   {mtx[1, 6]}"
    )
    print(
        f"\t\t      3 {mtx[2, 0]}   {mtx[2, 1]}   {mtx[2, 2]}  {mtx[2, 3]}  {mtx[2, 4]}   {mtx[2, 5]}   {mtx[2, 6]}"
    )
    print(
        f"\t\tTrue  4 {mtx[3, 0]}   {mtx[3, 1]}   {mtx[3, 2]}  {mtx[3, 3]}  {mtx[3, 4]}   {mtx[3, 5]}   {mtx[3, 6]}"
    )
    print(
        f"\t\t      5 {mtx[4, 0]}   {mtx[4, 1]}   {mtx[4, 2]}   {mtx[4, 3]}   {mtx[4, 4]}  {mtx[4, 5]}   {mtx[4, 6]}"
    )
    print(
        f"\t\t      6 {mtx[5, 0]}   {mtx[5, 1]}   {mtx[5, 2]}   {mtx[5, 3]}  {mtx[5, 4]} {mtx[5, 5]}  {mtx[5, 6]}"
    )
    print(
        f"\t\t      7 {mtx[6, 0]}   {mtx[6, 1]}   {mtx[6, 2]}   {mtx[6, 3]}   {mtx[6, 4]}   {mtx[6, 5]}   {mtx[6, 6]}"
    )


def print_parameters(classifier, parameters):

    if classifier == "Naive Bayes":
        return parameters
    if classifier == "kNN":
        return f"distance function = {parameters[0]}; nr neighbors = {parameters[1]}"
    if classifier == "Decision Tree":
        return f"criteria = {parameters[0]}; max depths = {parameters[1]}; min samples leaf = {parameters[2]}"
    if classifier == "Random Forest":
        return f"max features = {parameters[0]}; max depths = {parameters[1]}; nr estimators = {parameters[2]}"
    if classifier == "Gradient Boost":
        return f"max features = {parameters[0]}; max depths = {parameters[1]}; nr estimators = {parameters[2]}; learning rate = {parameters[3]}"
    if classifier == "XGBoost":
        return f"max depths = {parameters[0]}; nr estimators = {parameters[1]}"


def print_pre_parameters(pre_parameters):
    balanced = pre_parameters[0]
    normalized = pre_parameters[1]

    if balanced:
        params = "balanced "
    else:
        params = "unbalanced "
    if normalized:
        params += "and normalized"
    else:
        params += "and not normalized"
    return params


def print_analysis(reports, pre_parameters):
    pre_process_params = print_pre_parameters(pre_parameters)
    print(f"1. Applied preprocessing: {pre_process_params}\n")
    print("2. Classifiers:")
    for i in range(len(reports)):
        print(f"2.{i + 1} {reports[i][0]}")
        print(f"\t2.{i + 1}.1 Best accuracy")
        best_accuracy = reports[i][1]
        accuracy_parameters = print_parameters(reports[i][0], best_accuracy[0])
        print(f"\ta) Suggested parameterization: {accuracy_parameters}")
        print("\tb) Confusion matrix: ")
        print_cnf_mtx(best_accuracy[3])
        print(f"\t2.{i + 1}.2 Best specificity")
        best_specificity = reports[i][2]
        specifivity_parameters = print_parameters(reports[i][0], best_specificity[0])
        print(f"\ta) Suggested parameterization: {specifivity_parameters}")
        print("\tb) Confusion matrix: ")
        print_cnf_mtx(best_specificity[3])
    print("3. Comparative performance: NB | kNN | DT | RF | GB | XGB")
    accuracies = ""
    for report in reports:
        accuracies += "{} | ".format(f"{report[1][1]:.2f}")
    print("\t3.1 Accuracy: " + accuracies[:-2])
    specificities = ""
    for report in reports:
        specificities += "{} | ".format(f"{report[2][2]:.2f}")
    print("\t3.2 Specificity: " + specificities[:-2])


def print_analysis_CT(reports, pre_parameters):
    pre_process_params = print_pre_parameters(pre_parameters)
    print(f"1. Applied preprocessing: {pre_process_params}\n")
    print("2. Classifiers:")
    for i in range(len(reports)):
        clf_name = reports[i][0]
        report = reports[i][1]
        parameters = report[0]
        accuracy = report[1]
        cnf_mtx = report[2]
        print(f"2.{i + 1} {clf_name}")
        accuracy_parameters = print_parameters(clf_name, parameters)
        print(f"\ta) Suggested parameterization: {accuracy_parameters}")
        print("\tb) Accuracy: {}".format(f"{accuracy:.2f}"))
        print("\td) Confusion matrix: ")
        print_cnf_mtx_CT(cnf_mtx)

    print("3. Comparative performance: NB | kNN | DT | RF | GB | XGB")
    accuracies = ""
    for report in reports:
        accuracies += "{} | ".format(f"{report[1][1]:.2f}")
    print("\t3.1 Accuracy: " + accuracies[:-2])


def print_report(reports, pre_parameters):
    pre_process_params = print_pre_parameters(pre_parameters)
    print(f"1. Applied preprocessing: {pre_process_params}\n")
    print("2. Classifiers:")
    for i in range(len(reports)):
        report = reports[i]
        clf_name = report[0]
        parameters = report[1]
        accuracy = report[2]
        recall = report[3]
        cnf_mtx = report[4]
        print(f"2.{i + 1} {clf_name}")
        accuracy_parameters = print_parameters(clf_name, parameters)
        print(f"\ta) Suggested parameterization: {accuracy_parameters}")
        print("\tb) Accuracy: {}".format(f"{accuracy:.2f}"))
        print("\tc) Recall: {}".format(f"{recall:.2f}"))
        print("\td) Confusion matrix: ")
        print_cnf_mtx(cnf_mtx)

    print("3. Comparative performance: NB | kNN | DT | RF | GB | XGB")
    accuracies = ""
    for report in reports:
        accuracies += "{} | ".format(f"{report[2]:.2f}")
    print("\t3.1 Accuracy: " + accuracies[:-2])
    recalls = ""
    for report in reports:
        recalls += "{} | ".format(f"{report[3]:.2f}")
    print("\t3.2 Recall: " + recalls[:-2])
