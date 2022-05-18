# Load packages
library(shiny)
library(DT)
library(cvms)
library(ggimage)
library(tidyverse)
library(tidymodels)
library(textrecipes)
library(plotly)
library(patchwork)
library(DALEXtra)
library(tidytext)
library(shinycssloaders)





theme_set(theme_light())

### Code to bring DALEX into the shiny environment ###
new_dragon <- data.frame(
  year_of_birth = 200,
  height = 80,
  weight = 12.5,
  scars = 0,
  number_of_lost_teeth  = 5
)

model_lm <- lm(life_length ~ year_of_birth + height +
                 weight + scars + number_of_lost_teeth,
               data = dragons)

explainer_lm <- explain(model_lm,
                        data = dragons,
                        y = dragons$year_of_birth,
                        label = "model_lm")

bd_lm <- predict_parts_break_down(explainer_lm, new_observation = new_dragon)
### Just ignore this ######

# Sample pretty colors
colors = c("#FAB9ACFF", "#7BBC53FF", "#DE6736FF", "#67C1ECFF", "#E6B90DFF")

# Load data/objects
last_fit_object <- readRDS("last_fit_object")
xgb_wf <- readRDS("smote_boost_hnwf")
test <- readRDS("test")
train <- readRDS("train")
breakdown_vals <- readRDS("list_bd")
sample_set <- readRDS("sample_set") %>% 
  mutate(row = row_number(),
         select_input = paste("Respondent ", row, ": ", opinion_ban, sep = "")) %>% 
  mutate(alt_veh = str_replace_all(alt_veh, " ", "_"))

# Load VIP scores and SHAP scores
opinion_shap <- readRDS("opinion_shap")
vip_scores <- readRDS("vip_scores")

# Load sample survey
sample_survey <- readRDS("sample_survey")

# Alternative vehicles in case of ban
alt_veh <- c("ebike", "car", "bike","taxi", "walk", "lighttrain", "bus")


# Frequency of vehicle usage
freq_vehic =c("0", "1_5", "6_10", "11_15", "16_20", "more_20")

# Vector of predictors
vip_features <- xgb_wf %>% 
  extract_preprocessor() %>% 
  summary() %>% 
  filter(role == "predictor") %>% 
  pull(variable)

vip_train <- train %>%
  select(all_of(vip_features))

# Explainer for xgboost
explainer_xgboost <- explain_tidymodels(
  # a fitted workflow
  model <- xgb_wf,
  # Data should be passed without target column 
  # this shall be provided as the y argument
  data = vip_train,
  # y should be an integer
  y = as.integer(train$opinion_ban),
  label = "xg_boost",
  verbose = F
  
)

# Investigate factors driving model predictions
pred_prob <- xgb_wf %>%
  augment(vip_train %>%
            bind_cols(train %>% select(opinion_ban)))

# Calculate performance metrics
metrics <- last_fit_object %>% 
  collect_metrics() %>% 
  select(metric = .metric, score = .estimate)

# Extract predictions
pred_test <- last_fit_object %>% 
  collect_predictions() 

# Evaluate confusion matrix metrics
cmat <- pred_test %>% 
  mutate(across(c(.pred_class, opinion_ban), as.character)) %>% 
  evaluate(target_col = "opinion_ban",
           prediction_cols = ".pred_class",
           type = "binomial")

# Define UI for application that draws a histogram
ui <- navbarPage(
  title = HTML(paste("<b>","Model Explainability", "</b>")),
  
  tabPanel("Model performance",
           
           # Type of layout
           fluidRow(
             
             column(6,
                    h3(paste("Model evaluation metrics")),
                    br(),
                    tableOutput(outputId = "metrics") %>% withSpinner(color =
                                                                        sample(colors, 1))),
             column(
               6,
               h3(paste("Confusion Matrix")),
               h5("How many examples in each class were correctly/incorrectly?"),
               plotOutput(outputId = "cm") %>% withSpinner(color =
                                                             sample(colors, 1))
             )),
           
           fluidPage(
             column(6,
                    h3("ROC AUC Plot"),
                    h5("Ability to distinguish between classes"),
                    br(),
                    plotOutput(outputId = "auc") %>% withSpinner(color =
                                                                   sample(colors, 1))
                    
             ),
             column(6,
                    h3("Precision-Recall AUC Plot"),
                    h5("Tradeoff between precision and recall"),
                    br(),
                    plotOutput(outputId = "pr") %>% withSpinner(color =
                                                                  sample(colors, 1))
             )
           )
           
  ),# Model performance panel
  
  
  
  tabPanel(
    "Variable Importance",
    # Type of layout
    sidebarLayout(
      
      
      
      
      # Sidebar panel for inputs
      sidebarPanel(
        # Static HTML text
        HTML(paste("Variable Importance Plots are one way of
                 understanding which predictors have the
                 largest effect on the model outcomes.\n
                 
                 The main idea is to measure how much
                 a model's performance changes if the effect
                 of a selected explanatory variable,
                 or of a group of variables, is removed.
                 ")),
        br(), br(),
        
        sliderInput(inputId = "n_vip",
                    label = "Select the top important variables",
                    min = 1,
                    max = nrow(vip_scores),
                    value = 10)
        
      ),
      
      # Main panel for outputs
      mainPanel(
        plotOutput(outputId = "vip", width = "auto", height = "500px") %>% withSpinner(color =
                                                       sample(colors, 1)),
        #tableOutput(outputId = "vips")
        DTOutput(outputId = "survey") %>% withSpinner(color =
                                                        sample(colors, 1))
      )
    )
  ), # VIP panel,
  
  
  tabPanel(title = "Shapley Additive exPlanations (SHAP)",
           
           # Type of layout
           sidebarLayout(
             sidebarPanel(
               HTML(paste("<b>", "SHAP Summary Plot", "</b>")),
               # Static HTML explanation
               HTML(paste("How positively/negatively do the features affect model predictions?")),
               
               br(), br(),
               
               
               sliderInput(inputId = "n_shap",
                           label = "Select the top important variables",
                           min = 1,
                           max = nrow(opinion_shap %>% distinct(variable)),
                           value = 10),
               
               br(), br(),
               
               HTML(paste("<b>", "SHAP Dependence Plots", "</b>")),
               
               HTML(paste("Further investigate how each feature affects the model's predictions.")),
               
               
               # Select the predictor(s) to be investigated
               selectInput(inputId = "shap_dep",
                           label = "Select predictor(s) to investigate",
                           choices = unique(opinion_shap$variable),
                           selected = "dist_to_pub",
                           multiple = TRUE),
               
               
               
               br(),
               HTML(paste("It is optional to use a different variable for SHAP values on the y-axis, and color the points by the feature value of a designated variable.
                        This allows one to estimate the relationship between
                        one variable and another and how this affects predictions.")),
               # HTML(paste("It is also possible to investigate how the feature values of one predictor and the SHAP values
               # of another predictor relate to the model predictions.")),
               # 
               # Select x and y for dependence plots
               selectInput(inputId = "x",
                           label = "Feature value on X axis:",
                           choices = unique(opinion_shap$variable),
                           selected = "own_car",
                           multiple = FALSE),
               
               selectInput(inputId = "y",
                           label = "SHAP value on Y axis:",
                           choices = unique(opinion_shap$variable),
                           selected = "aware_ban_yes",
                           multiple = FALSE),
               # Will uncomment this if it proves useful in the future
               # selectInput(inputId = "c",
               #             label = "Color feature:",
               #             choices = unique(opinion_shap$variable),
               #             selected = "aware_ban_yes",
               #             multiple = FALSE),
               # 
               
             ),
             
             mainPanel(
               plotOutput(outputId = "shap_summary", inline = TRUE) %>% withSpinner(color = sample(colors, 1)),
               plotOutput(outputId = "shap_dependency") %>% withSpinner(color = sample(colors, 1)),
               plotOutput(outputId = "shap_dependency_int") %>% withSpinner(color = sample(colors, 1))
             )
           )
           
           
           
  ), # SHAP panel
  

  
  
  
  
  
  
  
  
  tabPanel("Local predictions",
           
           # Type of layout
           fluidRow(
             column(5,
                    h3("Select respondent"),
                    h5("Select an individual based on survey response"),
                    selectInput(inputId = "respondent", label = "",
                                choices = sample_set %>% pull(select_input), 
                                selected = (pull(sample_set, select_input))[1]),
                    DTOutput(outputId = "respondent_xtics") %>% withSpinner(color =
                                                                              sample(colors, 1))),
             
             column(7,
                    h3("Prediction"),
                    h5("Predicted class probabilities"),
                    br(),
                    tableOutput(outputId = "pred_table") %>% withSpinner(color =
                                                                           sample(colors, 1)),
                    br(),
                    plotlyOutput(outputId = "pie") %>% withSpinner(color =
                                                                     sample(colors, 1))
             )),
           hr(),
           
           fluidRow(
             h3("Break-down plots"),
             h5("Which variables contribute to this result the most?"),
             br(),
             
             HTML(paste("For this plots, due to the packages used, a probability of", "<b>", "0 - 0.5", "</b>", 
                        "represents", "<b>", "agree", "</b>.")),
             HTML(paste("<br/>","Purple/red variables increase the probability of agreeing/decrease probability of disagreeing with ban." )),
             
             
             
             plotOutput(outputId = "bd_plots", height = "500px") %>% withSpinner(color =
                                                                 sample(colors, 1))
             
           )
           
           
  ),# Local predictions panel
  
  # tabPanel(
  #   "shit",
  # sidebarLayout(
  #   sidebarPanel(width = 5,
  #     HTML(paste("How would individual predictions change if we altered
  #                some of the survey responses?")),
  #     
  #     fluidRow(
  #       column(6,
  #              paste("tay")),
  #       column(6,
  #             paste("Er"))
  #     ),
  #     
  #     
  #     
  #   ),
  #   mainPanel())
  # )
  
  
  tabPanel(
    title = "Scenario responses",
    h2(paste("How would individual predictions change if we altered
                 some of the survey responses?")),
    hr(),
    br(),
    
    
    # Type of layout
    fluidRow(
      column(2,
             h5("Select an individual based on survey response"),
             selectInput(inputId = "sc_respondent", label = "",
                         choices = sample_set %>% pull(select_input),
                         selected = (pull(sample_set, select_input))[1]),
      ),
      column(5,
             offset = 1,
             h3("Break-down plots"),
             h5("Which variables contribute to this result the most?"),
             
             br(),
             
             #HTML(paste("For this plots, due to the packages used, a probability of", "<b>", "0 - 0.5", "</b>", 
             #"represents", "<b>", "agree", "</b>.")),
             #HTML(paste("Purple/red variables increase the probability of agreeing/decrease probability of disagreeing with ban." )),
             
             
             checkboxInput("checkbox", label = "Make B-D plots?", value = F),
             plotOutput(outputId = "sc_bd_plots") %>% withSpinner(color = sample(colors, 1))),
      
      
      
      column(4,
             h3("Prediction"),
             h5("Predicted class probabilities"),
             br(),
             #tableOutput(outputId = "pred_table"),
             plotlyOutput(outputId = "sc_pie",

                          width = "auto",
                          height = "auto")%>% withSpinner(color =
                                                            sample(colors, 1))
             #br(),
             #tableOutput("sc_tbl")
      )),
      
      # column(4,
      #        plotlyOutput(outputId = "sc_pie",
      #                     
      #                     width = "auto",
      #                     height = "auto")%>% withSpinner(color =
      #                                                       sample(colors, 1)))),
    
    
    fluidRow(
      h3("Vary survey responses to simulate new scenarios"),
      # Aware ban
      column(3,
             selectInput(
               inputId = "sc_aware",
               label = "Awareness of ban:",
               choices = sample_set %>% 
                 distinct(aware_ban) %>% pull(aware_ban),
               selected = (pull(sample_set, select_input))[1]
             )),
      
      # Alternative vehicle
      column(3,
             selectInput(
               inputId = "sc_alt_vehic",
               label = "Alternative vehicle in case of ban:",
               choices = alt_veh,
               selected = (pull(sample_set, alt_veh))[1]
             )),
      
      # Frequency of car usage
      column(3,
             selectInput(
               inputId = "sc_freq_car",
               label = "Frequency of car usage:",
               choices = freq_vehic,
               selected = (pull(sample_set, freq_car))[1]
             )),
      
      # Frequency of bus usage
      column(
        3,
        selectInput(
          inputId = "sc_freq_bus",
          label = "Frequency of bus usage:",
          choices = freq_vehic,
          selected = (pull(sample_set, freq_bus))[1]
        ))
      
      
    ),
    
    fluidRow(
      
      # Distance to public transport (metres)
      column(3,
             sliderInput(
               inputId = "sc_dist_to_pub",
               label = "Distance to public transport (m):",
               min = 0, max = 8400,
               value = (pull(sample_set, dist_to_pub))[1]
             )),
      
      
      # Number of cars owned
      column(3,
             sliderInput(
               inputId = "sc_own_car",
               label = "Number of cars owned:",
               min = 0, max = 7,
               value = (pull(sample_set, own_car))[1]
             )),
      
      # Number of bikes owned
      column(3,
             sliderInput(
               inputId = "sc_own_bike",
               label = "Number of bikes owned:",
               min = 0, max = 7,
               value = (pull(sample_set, own_bike))[1]
             )),
      
      # Number of ebikes owned
      column(3,
             sliderInput(
               inputId = "sc_own_ebike",
               label = "Number of ebikes owned:",
               min = 0, max = 7,
               value = (pull(sample_set, own_ebike))[1]
             )))
    
    # fluidRow(
    #   column(6,
    #   h3("Break-down plots"),
    #   h5("Which variables contribute to this result the most?"),
    #   
    #   br(),
    #   
    #   #HTML(paste("For this plots, due to the packages used, a probability of", "<b>", "0 - 0.5", "</b>", 
    #              #"represents", "<b>", "agree", "</b>.")),
    #   #HTML(paste("Purple/red variables increase the probability of agreeing/decrease probability of disagreeing with ban." )),
    #   
    #   
    #   checkboxInput("checkbox", label = "Make B-D plots?", value = F),
    #   plotOutput(outputId = "sc_bd_plots") %>% withSpinner(color = sample(colors, 1)))
    #   
    #   )
    # 
    # 
    # )
    
    
    
  )  #what-if panel       
  
  
  
  
  
)

# Define server logic required map inputs to outputs
server <- function(input, output, session) {
  
  
  ##### Model Performance ###################################
  # Metrics table
  output$metrics = renderTable({
    metrics
  },
  striped = TRUE,
  spacing = "l",
  hover = TRUE,
  align = "lc",
  digits = 4,
  width = "90%",
  #caption = "Evaluation metrics."
  
  )
  
  # Confusion matrix
  output$cm = renderPlot({
    cmat$`Confusion Matrix` %>% pluck(1) %>% 
      plot_confusion_matrix(
        add_normalized = FALSE,
        palette = "Greens"
      )
  }
  )
  
  # Area under curve
  output$auc = renderPlot({
    auc_score <- roc_auc(pred_test, opinion_ban, .pred_agree) %>% 
      pull(.estimate)
    pred_test %>% 
      roc_curve(opinion_ban, .pred_agree) %>% 
      ggplot(mapping = aes(x = 1 - specificity, y = sensitivity)) +
      geom_abline(lty = 2, color = "gray80", size = 0.9) +
      geom_path(color = "dodgerblue", alpha = 0.6, size = 1.3) +
      coord_equal() +
      xlab("False Positive Rate") +
      ylab("True Positive Rate") +
      geom_label(aes(x = 0.5, y = 0.5),
                 label = paste("roc-auc-score:", round(auc_score ,2)))
  })
  
  
  # Precision-recall auc
  output$pr = renderPlot({
    pr_score <- pr_auc(pred_test, opinion_ban, .pred_agree) %>% 
      pull(.estimate)
    pred_test %>% 
      pr_curve(opinion_ban, .pred_agree) %>% 
      ggplot(mapping = aes(x = recall, y = precision)) +
      geom_path(color = "darkorange", alpha = 0.6, size = 1.3) +
      #coord_equal() +
      geom_label(aes(x = 0.5, y = 0.5),
                 label = paste("pr-auc-score:", round(pr_score ,2))) 
    
  })
  
  
  ##############################Model Performance#############
  
  
  
  
  
############ Variable importance plot #####################
  output$vip <- renderPlot({
    req(input$n_vip)
    vip_scores %>% 
      slice_head(n = input$n_vip) %>% 
      mutate(Variable = fct_reorder(Variable, Importance)) %>% 
      ggplot(mapping = aes(y = Variable, x = Importance)) +
      geom_point(size = 3, color = "dodgerblue") + 
      geom_segment(aes(y = Variable, yend = Variable, x = 0, xend = Importance), size = 2, color = "dodgerblue", alpha = 0.7 ) +
      ggtitle(paste("Top", input$n_vip, "variables influencing model performance.")) +
      theme(
        plot.title = element_text(hjust = 0.5)
      )
  })
  
  # Table Output
  output$survey <- renderDT({
    sample_survey %>% 
      slice_head(n = input$n_vip)
  },
  options = list(pageLength = 10)
  #caption = "Variable importance scores for the selected predictors in the model."
  )
  ############ Variable importance plot ##################### 
  
######### Shap summary plot ################################
  output$shap_summary <- renderPlot({
    req(input$n_shap)
    # Could be done with ggplot, right Kristina?
    # Top n variables
    vars = distinct(opinion_shap, variable) %>% 
      slice_head(n = input$n_shap) %>% pull(variable) %>% as.character()
    
    opinion_shap %>% 
      filter(variable %in% vars) %>% 
      mutate(variable = factor(variable, levels = vars)) %>% 
      shap.plot.summary()
    
  },
  width = 600, height = 600)
  
  # Shap dependency plots
  output$shap_dependency <- renderPlot({
    opinion_shap %>% 
      dplyr::filter(variable %in% input$shap_dep) %>% 
      ggplot(mapping = aes(x = rfvalue, y = value)) +
      geom_point(size = 0.5) +
      geom_smooth(method = "loess", se = F, color = "blue",
                  alpha = 0.7, lwd = 0.4) +
      facet_wrap(vars(variable), scales = "free_x") +
      ylab("SHAP") + xlab("feature value") +
      ggtitle("SHAP Dependency Plots") +
      theme(plot.title = element_text(hjust = 0.5))
  })
  
  
  # Colored SHAP dependency plots
  output$shap_dependency_int <- renderPlot({
    shap.plot.dependence(
      data_long = opinion_shap,
      x = input$x,
      y = input$y,
      color_feature = input$y,
      size0 = 1.2,
      smooth = T
    )
  })
######### Shap summary plot ################################  
  
 
  
  
  
  
  ##### Local predictions ##################################
  output$respondent_xtics = renderDT({
    sample_set %>% 
      filter(select_input == input$respondent) %>% 
      select(!contains( c(".pred", "row", "select_input"))) %>%
      mutate(across(everything(), as.character)) %>% 
      pivot_longer(everything(), 
                   names_to = "survey_questions", 
                   values_to = "response")
  },
  options = list(pageLength = 10))
  #caption = "Individual's responses")
  
  
  # Table predicted probabilities
  output$pred_table <- renderTable({
    plot_prob = sample_set %>% 
      filter(select_input == input$respondent) %>% 
      select(observed_label = opinion_ban, pred_agree = .pred_agree, pred_disagree = .pred_disagree)
  },
  striped = TRUE,
  spacing = "l",
  hover = TRUE,
  align = "lcr",
  digits = 2,
  width = "90%",
  #caption = "Evaluation metrics."
  )
  
  
  # Pie chart predicted probabilities
  output$pie = renderPlotly({
    plot_prob = sample_set %>% 
      filter(select_input == input$respondent) %>% 
      select(opinion_ban, pred_agree = .pred_agree, pred_disagree = .pred_disagree) %>% 
      pivot_longer(!opinion_ban, names_to = "predicted_class", values_to = "probabilities") %>%
      mutate(predicted_class = factor(predicted_class))
    
    plot_ly(plot_prob, labels = ~predicted_class,
            values = ~probabilities,
            type = 'pie', hole = 0.3, opacity = 0.8,
            textinfo = ~ paste("predicted_class + probabilities"),
            marker = list(colors = c("midnightblue", "#FA4616FF")))
  })
  
  
  # Breakdown plots
  output$bd_plots = renderPlot({
    # Bd based on shapley values
    shap_bd_plot = breakdown_vals %>% 
      pluck(str_extract(input$respondent, "[:digit:]") %>% as.integer()) %>% 
      pluck("shap_bd") %>% 
      group_by(variable) %>% 
      mutate(mean_val = mean(contribution)) %>% 
      ungroup() %>% 
      mutate(variable = fct_reorder(variable, abs(mean_val))) %>% 
      filter(contribution != 0) %>% 
      ggplot(mapping = aes(x = contribution, y = variable, fill = mean_val > 0)) +
      geom_boxplot() +
      geom_col(data = ~distinct(., variable, mean_val),
               aes(x = mean_val, y = variable),
               alpha = 0.5) +
      theme(legend.position = "none") +
      scale_fill_viridis_d() +
      labs(y = NULL)
    
    # Bd via sequential variable conditioning
    shap_bd_plot2 = breakdown_vals %>% 
      pluck(str_extract(input$respondent, "[:digit:]") %>% as.integer()) %>% 
      pluck("xgboost_breakdown") %>% 
      select(-row) %>% 
      plot(max_vars = 1000)
    
    # Patch things up
    
    shap_bd_plot + shap_bd_plot2
    
  })
  
################################ Local predictions ##########
  
  
  
  
   
  
#### Reactive scenario components in shiny ##################
  
  # Create an object that stores a reactive value x and y
  reactive_sc <- reactiveValues(x = NULL, y = NULL)
  observeEvent(input$sc_respondent,{
    # Observe change in respondent and assign this value to x
    reactive_sc$x <- input$sc_respondent
    
    # Update widgets based on this value
    # Aware ban
    updateSelectInput(session, inputId = "sc_aware",
                      selected = sample_set %>%
                        filter(select_input == reactive_sc$x) %>%
                        pull(aware_ban))
    
    # Alternative vehicle
    updateSelectInput(session, inputId = "sc_alt_vehic",
                      choices = c(alt_veh, sample_set %>%
                                    filter(select_input == reactive_sc$x) %>%
                                    pull(alt_veh)) %>% unique(),
                      selected = sample_set %>%
                        filter(select_input == reactive_sc$x) %>%
                        pull(alt_veh)) 
    # %>% 
    #   str_remove_all(" "))
    
    # Frequency of car
    updateSelectInput(session, inputId = "sc_freq_car",
                      selected = sample_set %>%
                        filter(select_input == reactive_sc$x) %>%
                        pull(freq_car))
    
    # Frequency of bus
    updateSelectInput(session, inputId = "sc_freq_bus",
                      selected = sample_set %>%
                        filter(select_input == reactive_sc$x) %>%
                        pull(freq_bus))
    
    # Distance to public transport
    updateSliderInput(session, inputId = "sc_dist_to_pub",
                      value = sample_set %>%
                        filter(select_input == reactive_sc$x) %>%
                        pull(dist_to_pub))
    
    # Number of cars owned
    updateSliderInput(session, inputId = "sc_own_car",
                      value = sample_set %>%
                        filter(select_input == reactive_sc$x) %>%
                        pull(own_car))
    
    # Number of bikes owned
    updateSliderInput(session, inputId = "sc_own_bike",
                      value = sample_set %>%
                        filter(select_input == reactive_sc$x) %>%
                        pull(own_bike))
    
    # Number of ebikes owned
    updateSliderInput(session, inputId = "sc_own_ebike",
                      value = sample_set %>%
                        filter(select_input == reactive_sc$x) %>%
                        pull(own_ebike))
    
    # Dynamic tibble
    # sc_sample_set <- sample_set %>% 
    #   filter(select_input == reactive_sc$x) %>% 
    #   mutate(aware_ban = input$sc_aware,
    #          alt_veh = input$sc_alt_vehic,
    #          freq_car = input$sc_freq_car,
    #          freq_bus = input$sc_freq_bus,
    #          dist_to_pub = input$sc_dist_to_pub,
    #          own_car = input$sc_own_car,
    #          own_bike = input$sc_own_bike,
    #          own_ebike = input$sc_own_ebike)
    # 
    # reactive_sc$y <- sc_sample_set %>% 
    #   mutate(alt_veh = str_replace_all(alt_veh, "_", " ")) %>% 
    #   select(-contains(".pred"))
    
    
    # Dynamic tibble
    output$sc_tbl = renderTable({
      req(reactive_sc$x)
      # Sys.sleep(1)
      sc_sample_set <- sample_set %>%
        filter(select_input == reactive_sc$x) %>%
        mutate(aware_ban = input$sc_aware,
               alt_veh = input$sc_alt_vehic,
               freq_car = input$sc_freq_car,
               freq_bus = input$sc_freq_bus,
               dist_to_pub = input$sc_dist_to_pub,
               own_car = input$sc_own_car,
               own_bike = input$sc_own_bike,
               own_ebike = input$sc_own_ebike)
      #mutate(alt_veh = str_remove_all(alt_veh, " ")) %>%
      
      # Store dataset in variable y
      reactive_sc$y <- sc_sample_set %>%
        mutate(alt_veh = str_replace_all(alt_veh, "_", " ")) %>%
        select(-contains(".pred"))
      
      sc_sample_set %>% 
        select(aware_ban, alt_veh, freq_car,
               freq_bus, dist_to_pub,
               own_car, own_bike, own_ebike) %>% 
        mutate(across(everything(), as.character)) %>% 
        pivot_longer(everything(), names_to = "features",
                     values_to =  "values")
      
      
    })
    
    # Replace _ with " "
    
    
    # Make predictions on scenario sample set
    output$sc_pie = renderPlotly({
      sc_sample_set <- sample_set %>%
        filter(select_input == reactive_sc$x) %>%
        mutate(aware_ban = input$sc_aware,
               alt_veh = input$sc_alt_vehic,
               freq_car = input$sc_freq_car,
               freq_bus = input$sc_freq_bus,
               dist_to_pub = input$sc_dist_to_pub,
               own_car = input$sc_own_car,
               own_bike = input$sc_own_bike,
               own_ebike = input$sc_own_ebike)
      #mutate(alt_veh = str_remove_all(alt_veh, " ")) %>%
      
      # Store dataset in variable y
      reactive_sc$y <- sc_sample_set %>%
        mutate(alt_veh = str_replace_all(alt_veh, "_", " ")) %>%
        select(-contains(".pred"))
      #req(reactive_sc$y)
      plot_prob = augment(xgb_wf, reactive_sc$y) %>%
        select(opinion_ban, pred_agree = .pred_agree, pred_disagree = .pred_disagree) %>%
        pivot_longer(!opinion_ban, names_to = "predicted_class", values_to = "probabilities") %>%
        mutate(predicted_class = factor(predicted_class))
      
      plot_ly(plot_prob, labels = ~predicted_class,
              values = ~probabilities,
              type = 'pie', hole = 0.3, opacity = 0.8,
              textinfo = ~ paste("predicted_class + probabilities"),
              marker = list(colors = c("midnightblue", "#FA4616FF")))
      
      
    })
    
    
    # Break down plots based on scenario values via
    # sequential variable conditioning
    output$sc_bd_plots = renderPlot({
      if(input$checkbox == TRUE && !identical(reactive_sc$y, sample_set %>%
                                              filter(select_input == reactive_sc$x) %>% 
                                              mutate(alt_veh = str_replace_all(alt_veh, "_", " ")) %>%
                                              select(-contains(".pred")))){
        xgboost_breakdown = predict_parts(
          explainer = explainer_xgboost,
          new_observation = reactive_sc$y,
          order = breakdown_vals %>%
            pluck(str_extract(reactive_sc$x, "[:digit:]") %>% as.integer()) %>%
            pluck("xgboost_breakdown") %>%
            select(variable_name) %>%
            filter(!variable_name %in% c("intercept", "")) %>%
            pull(variable_name))
        
        plot(xgboost_breakdown, max_vars = 1000)
      }else{
        
        breakdown_vals %>% 
          pluck(str_extract(reactive_sc$x, "[:digit:]") %>% as.integer()) %>% 
          pluck("xgboost_breakdown") %>% 
          select(-row) %>% 
          plot(max_vars = 1000)
        
      }
      
      
      
    })
    
    
    
  })
  
  # output$sc_pie = renderTable({
  #   #req(reactive_sc$y)
  #   
  #   sc_sample_set <- augment(xgb_wf, sample_set %>%
  #     filter(select_input == reactive_sc$x) %>%
  #     mutate(aware_ban = input$sc_aware,
  #            alt_veh = input$sc_alt_vehic,
  #            freq_car = input$sc_freq_car,
  #            freq_bus = input$sc_freq_bus,
  #            dist_to_pub = input$sc_dist_to_pub,
  #            own_car = input$sc_own_car,
  #            own_bike = input$sc_own_bike,
  #            own_ebike = input$sc_own_ebike) %>% 
  #     mutate(alt_veh = str_replace_all(alt_veh, "_", " ")) %>%
  #     select(-contains(".pred"))) %>% 
  #     select(aware_ban, alt_veh, freq_car,
  #            freq_bus, dist_to_pub,
  #            own_car, own_bike, own_ebike, contains(".pred")) %>% 
  #     mutate(across(everything(), as.character)) %>% 
  #     pivot_longer(everything(), names_to = "features",
  #                                  values_to =  "values")
  #   
  #   # sc_sample_set %>%
  #   #   select(aware_ban, alt_veh, freq_car,
  #   #          freq_bus, dist_to_pub,
  #   #          own_car, own_bike, own_ebike) %>%
  #   #   mutate(across(everything(), as.character)) %>%
  #   #   pivot_longer(everything(), names_to = "features",
  #   #                values_to =  "values")
  #   
  #   # augment(xgb_wf, sc_sample_set) %>% 
  #   #   select(contains(".pred"))
  #   
  # # plot_prob = augment(xgb_wf, sc_sample_set) %>%
  # #   select(opinion_ban, pred_agree = .pred_agree, pred_disagree = .pred_disagree) %>%
  # #   pivot_longer(!opinion_ban, names_to = "predicted_class", values_to = "probabilities") %>%
  # #   mutate(predicted_class = factor(predicted_class))
  # # 
  # # plot_ly(plot_prob, labels = ~predicted_class,
  # #         values = ~probabilities,
  # #         type = 'pie', hole = 0.3, opacity = 0.8,
  # #         textinfo = ~ paste("predicted_class + probabilities"),
  # #         marker = list(colors = c("midnightblue", "#FA4616FF")))
  # 
  # 
  # })
  
  
  #output$tay <- renderText(div_ru())
  
  # observeEvent({
  #   r$x = input$sc_respondent
  # #   #x = div_ru()
  # #   #updateSelectInput(session, inputId = "sc_respondent",  = input$x2)
  #   # updateSelectInput(session, inputId = "sc_aware",
  #   #                   selected = sample_set %>%
  #   #                       filter(select_input == renderText(div_ru())) %>%
  #   #                       pull(aware_ban))
  # })
  
  
  # observeEvent(input$sc_respondent,
  #              {
  #                text_reactive$text <- input$sc_respondent
  #              })
  # 
  # 
  # # reactiveValues
  # text_reactive <- reactiveValues(
  #   text = "No text has been submitted yet."
  # )
  
  # Selected<-reactiveValues(sc_respondent=NULL)
  # 
  # 
  # 
  # observeEvent(input$sc_respondent, Selected$sc_respondent<-(input$sc_respondent))
  # 
 
  #### Reactive scenario components in shiny ################## 
  

  
}

# Run the application 
shinyApp(ui = ui, server = server)
