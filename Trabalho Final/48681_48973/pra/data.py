import face_recognition as fr

# Função que serve para detetar o rosto numa imagem e codifica-la retornado True ser algum for encontrado
def recognize(url_picture):
    picture = fr.load_image_file(url_picture)
    users_detetion = fr.face_encodings(picture)
    if(len(users_detetion) > 0):
        return True, users_detetion
    
    return False, []

# Função que serve para armazenar e retornar os rostos conhecidos da base de dados
def get_users():
    users_database = []
    users_name_database = []

    user1 = recognize("./img/diogo_mestre.jpeg")

    if(user1[0]):
        users_database.append(user1[1][0])
        users_name_database.append("Diogo Mestre")
        
    user2 = recognize("./img/rodrigo_alves.png")

    if(user2[0]):
        users_database.append(user2[1][0])
        users_name_database.append("Rodrigo Alves")
        
    return users_database, users_name_database