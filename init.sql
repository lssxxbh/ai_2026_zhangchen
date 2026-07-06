CREATE DATABASE IF NOT EXISTS medical_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE medical_ai;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS conversations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    title VARCHAR(200) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS chat_messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT NOT NULL,
    role ENUM('user', 'assistant') NOT NULL,
    message TEXT NOT NULL,
    json_result LONGTEXT,
    file_name VARCHAR(255),
    file_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    INDEX idx_conversation_id (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;




USE medical_ai;

-- 医学知识图谱-节点主表
CREATE TABLE `medical_node` (
                                `id` VARCHAR(20) NOT NULL COMMENT '节点唯一ID，原文id',
                                `name` VARCHAR(500) NOT NULL COMMENT '节点名称',
                                `parent_id` VARCHAR(20) NULL DEFAULT NULL COMMENT '父节点ID，顶层系统为null',
                                `level` TINYINT NOT NULL COMMENT '层级：1系统/2子类/3疾病/4附属(症状/风险/治疗)',
                                `node_type` VARCHAR(30) NOT NULL COMMENT '原始type：system/category/disease/symptom/risk_factor/therapy',
                                `category` VARCHAR(20) NOT NULL COMMENT '中文分类：系统/子类/疾病/症状/危险因素/治疗',
                                `english` VARCHAR(500) NULL DEFAULT NULL COMMENT '英文名称/病名',
                                `description` TEXT NULL DEFAULT NULL COMMENT '疾病描述文本',
                                `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '入库时间',
                                `update_time` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                                PRIMARY KEY (`id`),
                                INDEX idx_parent (`parent_id`),
                                INDEX idx_level (`level`),
                                INDEX idx_node_type (`node_type`),
                                INDEX idx_name (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='医学知识图谱-全部节点';


-- 医学知识图谱-关系边表
CREATE TABLE `medical_relation` (
                                    `rid` BIGINT AUTO_INCREMENT NOT NULL COMMENT '关系自增主键',
                                    `source_id` VARCHAR(20) NOT NULL COMMENT '子节点ID',
                                    `target_id` VARCHAR(20) NOT NULL COMMENT '父节点ID',
                                    `rel_type` VARCHAR(50) NOT NULL COMMENT '关系类型',
                                    `rel_desc` VARCHAR(200) NULL COMMENT '关系中文说明',
                                    `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP,
                                    PRIMARY KEY (`rid`),
                                    UNIQUE KEY uk_source_target (`source_id`,`target_id`) COMMENT '避免重复父子关系',
                                    INDEX idx_source (`source_id`),
                                    INDEX idx_target (`target_id`),
                                    INDEX idx_rel_type (`rel_type`),
                                    FOREIGN KEY (`source_id`) REFERENCES medical_node(`id`) ON DELETE CASCADE,
                                    FOREIGN KEY (`target_id`) REFERENCES medical_node(`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='医学知识图谱-关联关系';


