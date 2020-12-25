import sys
class FaceRecognitionCredentials():
    EMP_ATTENDANCE_IDENTIFY_MSSQL_IP_ADDRESS = ''
    EMP_ATTENDANCE_IDENTIFY_MSSQL_DATABASE = ''
    # EMP_ATTENDANCE_IDENTIFY_MSSQL_USER_NAME = ''
    # EMP_ATTENDANCE_IDENTIFY_MSSQL_PASSWORD = ''
    DATASET_FOLDER_NAME = ''
    def loadConfigurations(requestInput):
        try:
            FaceRecognitionCredentials.EMP_ATTENDANCE_IDENTIFY_MSSQL_IP_ADDRESS = requestInput.args['EMP_ATTENDANCE_IDENTIFY_MSSQL_IP_ADDRESS']
            FaceRecognitionCredentials.EMP_ATTENDANCE_IDENTIFY_MSSQL_DATABASE = requestInput.args['EMP_ATTENDANCE_IDENTIFY_MSSQL_DATABASE']
            # FaceRecognitionCredentials.EMP_ATTENDANCE_IDENTIFY_MSSQL_USER_NAME = requestInput.args['EMP_ATTENDANCE_IDENTIFY_MSSQL_USER_NAME']
            # FaceRecognitionCredentials.EMP_ATTENDANCE_IDENTIFY_MSSQL_PASSWORD = requestInput.args['EMP_ATTENDANCE_IDENTIFY_MSSQL_PASSWORD']
            FaceRecognitionCredentials.DATASET_FOLDER_NAME = requestInput.args['DATASET_FOLDER_NAME']
            return 'success'
        except:
             return sys.exc_info()[1]