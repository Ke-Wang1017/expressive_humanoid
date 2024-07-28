/*
 * Interface wrapper code.
 *
 * Generated by SIP 4.19.3
 *
 * Copyright (C) 2009 Autodesk, Inc. and/or its licensors.
 * All Rights Reserved.
 * 
 * The coded instructions, statements, computer programs, and/or related material 
 * (collectively the "Data") in these files contain unpublished information 
 * proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 * Canada and United States of America federal copyright law and by international 
 * treaties. 
 * 
 * The Data may not be disclosed or distributed to third parties, in whole or in
 * part, without the prior written consent of Autodesk, Inc. ("Autodesk").
 * 
 * THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 * ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 * WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 * BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 * NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 * WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 * OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 * 
 * IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 * OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 * OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 * SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 * OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 * HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 * FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 * ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 * WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 * OR DAMAGE. 
 */

#include "sipAPIfbx.h"


#line 5 "/home/ke/Documents/expressive-humanoid/sip/fbxusernotification.sip"
#include <fbxsdk.h>
#line 44 "/home/ke/Documents/expressive-humanoid/build/Python38_x64/sipfbxFbxManager.cpp"


class sipFbxManager : public  ::FbxManager
{
public:
    sipFbxManager();
    virtual ~sipFbxManager();

    /*
     * There is a protected method for every virtual method visible from
     * this class.
     */
protected:
    void SetIOSettings( ::FbxIOSettings*);
     ::FbxIOSettings* GetIOSettings();
    void Destroy();

public:
    sipSimpleWrapper *sipPySelf;

private:
    sipFbxManager(const sipFbxManager &);
    sipFbxManager &operator = (const sipFbxManager &);

    char sipPyMethods[3];
};

sipFbxManager::sipFbxManager():  ::FbxManager(), sipPySelf(0)
{
    memset(sipPyMethods, 0, sizeof (sipPyMethods));
}

sipFbxManager::~sipFbxManager()
{
    sipInstanceDestroyed(sipPySelf);
}

void sipFbxManager::SetIOSettings( ::FbxIOSettings*a0)
{
    sip_gilstate_t sipGILState;
    PyObject *sipMeth;

    sipMeth = sipIsPyMethod(&sipGILState,&sipPyMethods[0],sipPySelf,NULL,sipName_SetIOSettings);

    if (!sipMeth)
    {
         ::FbxManager::SetIOSettings(a0);
        return;
    }

    extern void sipVH_fbx_3(sip_gilstate_t, sipVirtErrorHandlerFunc, sipSimpleWrapper *, PyObject *,  ::FbxIOSettings*);

    sipVH_fbx_3(sipGILState, 0, sipPySelf, sipMeth, a0);
}

 ::FbxIOSettings* sipFbxManager::GetIOSettings()
{
    sip_gilstate_t sipGILState;
    PyObject *sipMeth;

    sipMeth = sipIsPyMethod(&sipGILState,&sipPyMethods[1],sipPySelf,NULL,sipName_GetIOSettings);

    if (!sipMeth)
        return  ::FbxManager::GetIOSettings();

    extern  ::FbxIOSettings* sipVH_fbx_2(sip_gilstate_t, sipVirtErrorHandlerFunc, sipSimpleWrapper *, PyObject *);

    return sipVH_fbx_2(sipGILState, 0, sipPySelf, sipMeth);
}

void sipFbxManager::Destroy()
{
    sip_gilstate_t sipGILState;
    PyObject *sipMeth;

    sipMeth = sipIsPyMethod(&sipGILState,&sipPyMethods[2],sipPySelf,NULL,sipName_Destroy);

    if (!sipMeth)
    {
         ::FbxManager::Destroy();
        return;
    }

    extern void sipVH_fbx_0(sip_gilstate_t, sipVirtErrorHandlerFunc, sipSimpleWrapper *, PyObject *);

    sipVH_fbx_0(sipGILState, 0, sipPySelf, sipMeth);
}


PyDoc_STRVAR(doc_FbxManager_Create, "Create() -> FbxManager");

extern "C" {static PyObject *meth_FbxManager_Create(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_Create(PyObject *, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        if (sipParseArgs(&sipParseErr, sipArgs, ""))
        {
             ::FbxManager*sipRes;

            sipRes =  ::FbxManager::Create();

            return sipConvertFromType(sipRes,sipType_FbxManager,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_Create, doc_FbxManager_Create);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_Destroy, "Destroy(self)");

extern "C" {static PyObject *meth_FbxManager_Destroy(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_Destroy(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;
    bool sipSelfWasArg = (!sipSelf || sipIsDerivedClass((sipSimpleWrapper *)sipSelf));

    {
         ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxManager, &sipCpp))
        {
            (sipSelfWasArg ? sipCpp-> ::FbxManager::Destroy() : sipCpp->Destroy());

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_Destroy, doc_FbxManager_Destroy);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_GetFileFormatVersion, "GetFileFormatVersion() -> Tuple[int, int, int]");

extern "C" {static PyObject *meth_FbxManager_GetFileFormatVersion(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_GetFileFormatVersion(PyObject *, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
        int a1;
        int a2;

        if (sipParseArgs(&sipParseErr, sipArgs, ""))
        {
             ::FbxManager::GetFileFormatVersion(a0,a1,a2);

            return sipBuildResult(0,"(iii)",a0,a1,a2);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_GetFileFormatVersion, doc_FbxManager_GetFileFormatVersion);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_CreateNewObjectFromClassId, "CreateNewObjectFromClassId(self, FbxClassId, str, FbxObject = None, FbxObject = None) -> FbxObject");

extern "C" {static PyObject *meth_FbxManager_CreateNewObjectFromClassId(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_CreateNewObjectFromClassId(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxClassId* a0;
        const char* a1;
        PyObject *a1Keep;
         ::FbxObject* a2 = 0;
        const  ::FbxObject* a3 = 0;
         ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9A8|J8J8", &sipSelf, sipType_FbxManager, &sipCpp, sipType_FbxClassId, &a0, &a1Keep, &a1, sipType_FbxObject, &a2, sipType_FbxObject, &a3))
        {
             ::FbxObject*sipRes;

            sipRes = sipCpp->CreateNewObjectFromClassId(*a0,a1,a2,a3);
            Py_DECREF(a1Keep);

            return sipConvertFromType(sipRes,sipType_FbxObject,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_CreateNewObjectFromClassId, doc_FbxManager_CreateNewObjectFromClassId);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_FindClass, "FindClass(self, str) -> FbxClassId");

extern "C" {static PyObject *meth_FbxManager_FindClass(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_FindClass(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const char* a0;
        PyObject *a0Keep;
        const  ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8", &sipSelf, sipType_FbxManager, &sipCpp, &a0Keep, &a0))
        {
             ::FbxClassId*sipRes;

            sipRes = new  ::FbxClassId(sipCpp->FindClass(a0));
            Py_DECREF(a0Keep);

            return sipConvertFromNewType(sipRes,sipType_FbxClassId,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_FindClass, doc_FbxManager_FindClass);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_FindFbxFileClass, "FindFbxFileClass(self, str, str) -> FbxClassId");

extern "C" {static PyObject *meth_FbxManager_FindFbxFileClass(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_FindFbxFileClass(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const char* a0;
        PyObject *a0Keep;
        const char* a1;
        PyObject *a1Keep;
        const  ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8A8", &sipSelf, sipType_FbxManager, &sipCpp, &a0Keep, &a0, &a1Keep, &a1))
        {
             ::FbxClassId*sipRes;

            sipRes = new  ::FbxClassId(sipCpp->FindFbxFileClass(a0,a1));
            Py_DECREF(a0Keep);
            Py_DECREF(a1Keep);

            return sipConvertFromNewType(sipRes,sipType_FbxClassId,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_FindFbxFileClass, doc_FbxManager_FindFbxFileClass);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_CreateDataType, "CreateDataType(self, str, EFbxType) -> FbxDataType");

extern "C" {static PyObject *meth_FbxManager_CreateDataType(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_CreateDataType(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const char* a0;
        PyObject *a0Keep;
         ::EFbxType a1;
         ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8E", &sipSelf, sipType_FbxManager, &sipCpp, &a0Keep, &a0, sipType_EFbxType, &a1))
        {
             ::FbxDataType*sipRes;

            sipRes = new  ::FbxDataType(sipCpp->CreateDataType(a0,a1));
            Py_DECREF(a0Keep);

            return sipConvertFromNewType(sipRes,sipType_FbxDataType,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_CreateDataType, doc_FbxManager_CreateDataType);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_GetDataTypeFromName, "GetDataTypeFromName(self, str) -> FbxDataType");

extern "C" {static PyObject *meth_FbxManager_GetDataTypeFromName(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_GetDataTypeFromName(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const char* a0;
        PyObject *a0Keep;
         ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8", &sipSelf, sipType_FbxManager, &sipCpp, &a0Keep, &a0))
        {
             ::FbxDataType*sipRes;

            sipRes = new  ::FbxDataType(sipCpp->GetDataTypeFromName(a0));
            Py_DECREF(a0Keep);

            return sipConvertFromNewType(sipRes,sipType_FbxDataType,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_GetDataTypeFromName, doc_FbxManager_GetDataTypeFromName);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_GetDataTypeCount, "GetDataTypeCount(self) -> int");

extern "C" {static PyObject *meth_FbxManager_GetDataTypeCount(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_GetDataTypeCount(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxManager, &sipCpp))
        {
            int sipRes;

            sipRes = sipCpp->GetDataTypeCount();

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_GetDataTypeCount, doc_FbxManager_GetDataTypeCount);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_GetDataType, "GetDataType(self, int) -> FbxDataType");

extern "C" {static PyObject *meth_FbxManager_GetDataType(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_GetDataType(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bi", &sipSelf, sipType_FbxManager, &sipCpp, &a0))
        {
             ::FbxDataType*sipRes;

            sipRes = &sipCpp->GetDataType(a0);

            return sipConvertFromType(sipRes,sipType_FbxDataType,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_GetDataType, doc_FbxManager_GetDataType);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_GetUserNotification, "GetUserNotification(self) -> FbxUserNotification");

extern "C" {static PyObject *meth_FbxManager_GetUserNotification(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_GetUserNotification(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxManager, &sipCpp))
        {
             ::FbxUserNotification*sipRes;

            sipRes = sipCpp->GetUserNotification();

            return sipConvertFromType(sipRes,sipType_FbxUserNotification,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_GetUserNotification, doc_FbxManager_GetUserNotification);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_SetUserNotification, "SetUserNotification(self, FbxUserNotification)");

extern "C" {static PyObject *meth_FbxManager_SetUserNotification(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_SetUserNotification(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxUserNotification* a0;
         ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8", &sipSelf, sipType_FbxManager, &sipCpp, sipType_FbxUserNotification, &a0))
        {
            sipCpp->SetUserNotification(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_SetUserNotification, doc_FbxManager_SetUserNotification);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_GetIOSettings, "GetIOSettings(self) -> FbxIOSettings");

extern "C" {static PyObject *meth_FbxManager_GetIOSettings(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_GetIOSettings(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;
    bool sipSelfWasArg = (!sipSelf || sipIsDerivedClass((sipSimpleWrapper *)sipSelf));

    {
         ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxManager, &sipCpp))
        {
             ::FbxIOSettings*sipRes;

            sipRes = (sipSelfWasArg ? sipCpp-> ::FbxManager::GetIOSettings() : sipCpp->GetIOSettings());

            return sipConvertFromType(sipRes,sipType_FbxIOSettings,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_GetIOSettings, doc_FbxManager_GetIOSettings);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_SetIOSettings, "SetIOSettings(self, FbxIOSettings)");

extern "C" {static PyObject *meth_FbxManager_SetIOSettings(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_SetIOSettings(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;
    bool sipSelfWasArg = (!sipSelf || sipIsDerivedClass((sipSimpleWrapper *)sipSelf));

    {
         ::FbxIOSettings* a0;
         ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8", &sipSelf, sipType_FbxManager, &sipCpp, sipType_FbxIOSettings, &a0))
        {
            (sipSelfWasArg ? sipCpp-> ::FbxManager::SetIOSettings(a0) : sipCpp->SetIOSettings(a0));

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_SetIOSettings, doc_FbxManager_SetIOSettings);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_GetIOPluginRegistry, "GetIOPluginRegistry(self) -> FbxIOPluginRegistry");

extern "C" {static PyObject *meth_FbxManager_GetIOPluginRegistry(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_GetIOPluginRegistry(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxManager, &sipCpp))
        {
             ::FbxIOPluginRegistry*sipRes;

            sipRes = sipCpp->GetIOPluginRegistry();

            return sipConvertFromType(sipRes,sipType_FbxIOPluginRegistry,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_GetIOPluginRegistry, doc_FbxManager_GetIOPluginRegistry);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_LoadPluginsDirectory, "LoadPluginsDirectory(self, str, str) -> bool");

extern "C" {static PyObject *meth_FbxManager_LoadPluginsDirectory(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_LoadPluginsDirectory(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        char* a0;
        PyObject *a0Keep;
        char* a1;
        PyObject *a1Keep;
         ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8A8", &sipSelf, sipType_FbxManager, &sipCpp, &a0Keep, &a0, &a1Keep, &a1))
        {
            bool sipRes;

            sipRes = sipCpp->LoadPluginsDirectory(a0,a1);
            Py_DECREF(a0Keep);
            Py_DECREF(a1Keep);

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_LoadPluginsDirectory, doc_FbxManager_LoadPluginsDirectory);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_LoadPlugin, "LoadPlugin(self, str) -> bool");

extern "C" {static PyObject *meth_FbxManager_LoadPlugin(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_LoadPlugin(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        char* a0;
        PyObject *a0Keep;
         ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BA8", &sipSelf, sipType_FbxManager, &sipCpp, &a0Keep, &a0))
        {
            bool sipRes;

            sipRes = sipCpp->LoadPlugin(a0);
            Py_DECREF(a0Keep);

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_LoadPlugin, doc_FbxManager_LoadPlugin);

    return NULL;
}


PyDoc_STRVAR(doc_FbxManager_UnloadPlugins, "UnloadPlugins(self) -> bool");

extern "C" {static PyObject *meth_FbxManager_UnloadPlugins(PyObject *, PyObject *);}
static PyObject *meth_FbxManager_UnloadPlugins(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxManager *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxManager, &sipCpp))
        {
            bool sipRes;

            sipRes = sipCpp->UnloadPlugins();

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxManager, sipName_UnloadPlugins, doc_FbxManager_UnloadPlugins);

    return NULL;
}


/* Call the instance's destructor. */
extern "C" {static void release_FbxManager(void *, int);}
static void release_FbxManager(void *sipCppV, int sipState)
{
    if (sipState & SIP_DERIVED_CLASS)
        delete reinterpret_cast<sipFbxManager *>(sipCppV);
}


extern "C" {static void dealloc_FbxManager(sipSimpleWrapper *);}
static void dealloc_FbxManager(sipSimpleWrapper *sipSelf)
{
    if (sipIsDerivedClass(sipSelf))
        reinterpret_cast<sipFbxManager *>(sipGetAddress(sipSelf))->sipPySelf = NULL;

    if (sipIsOwnedByPython(sipSelf))
    {
        release_FbxManager(sipGetAddress(sipSelf), sipIsDerivedClass(sipSelf));
    }
}


extern "C" {static void *init_type_FbxManager(sipSimpleWrapper *, PyObject *, PyObject *, PyObject **, PyObject **, PyObject **);}
static void *init_type_FbxManager(sipSimpleWrapper *sipSelf, PyObject *sipArgs, PyObject *sipKwds, PyObject **sipUnused, PyObject **, PyObject **sipParseErr)
{
    sipFbxManager *sipCpp = 0;

    {
        if (sipParseKwdArgs(sipParseErr, sipArgs, sipKwds, NULL, sipUnused, ""))
        {
            sipCpp = new sipFbxManager();

            sipCpp->sipPySelf = sipSelf;

            return sipCpp;
        }
    }

    return NULL;
}


static PyMethodDef methods_FbxManager[] = {
    {SIP_MLNAME_CAST(sipName_Create), meth_FbxManager_Create, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_Create)},
    {SIP_MLNAME_CAST(sipName_CreateDataType), meth_FbxManager_CreateDataType, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_CreateDataType)},
    {SIP_MLNAME_CAST(sipName_CreateNewObjectFromClassId), meth_FbxManager_CreateNewObjectFromClassId, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_CreateNewObjectFromClassId)},
    {SIP_MLNAME_CAST(sipName_Destroy), meth_FbxManager_Destroy, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_Destroy)},
    {SIP_MLNAME_CAST(sipName_FindClass), meth_FbxManager_FindClass, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_FindClass)},
    {SIP_MLNAME_CAST(sipName_FindFbxFileClass), meth_FbxManager_FindFbxFileClass, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_FindFbxFileClass)},
    {SIP_MLNAME_CAST(sipName_GetDataType), meth_FbxManager_GetDataType, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_GetDataType)},
    {SIP_MLNAME_CAST(sipName_GetDataTypeCount), meth_FbxManager_GetDataTypeCount, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_GetDataTypeCount)},
    {SIP_MLNAME_CAST(sipName_GetDataTypeFromName), meth_FbxManager_GetDataTypeFromName, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_GetDataTypeFromName)},
    {SIP_MLNAME_CAST(sipName_GetFileFormatVersion), meth_FbxManager_GetFileFormatVersion, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_GetFileFormatVersion)},
    {SIP_MLNAME_CAST(sipName_GetIOPluginRegistry), meth_FbxManager_GetIOPluginRegistry, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_GetIOPluginRegistry)},
    {SIP_MLNAME_CAST(sipName_GetIOSettings), meth_FbxManager_GetIOSettings, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_GetIOSettings)},
    {SIP_MLNAME_CAST(sipName_GetUserNotification), meth_FbxManager_GetUserNotification, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_GetUserNotification)},
    {SIP_MLNAME_CAST(sipName_LoadPlugin), meth_FbxManager_LoadPlugin, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_LoadPlugin)},
    {SIP_MLNAME_CAST(sipName_LoadPluginsDirectory), meth_FbxManager_LoadPluginsDirectory, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_LoadPluginsDirectory)},
    {SIP_MLNAME_CAST(sipName_SetIOSettings), meth_FbxManager_SetIOSettings, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_SetIOSettings)},
    {SIP_MLNAME_CAST(sipName_SetUserNotification), meth_FbxManager_SetUserNotification, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_SetUserNotification)},
    {SIP_MLNAME_CAST(sipName_UnloadPlugins), meth_FbxManager_UnloadPlugins, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxManager_UnloadPlugins)}
};

PyDoc_STRVAR(doc_FbxManager, "\1FbxManager()");


sipClassTypeDef sipTypeDef_fbx_FbxManager = {
    {
        -1,
        0,
        0,
        SIP_TYPE_CLASS,
        sipNameNr_FbxManager,
        {0},
        0
    },
    {
        sipNameNr_FbxManager,
        {0, 0, 1},
        18, methods_FbxManager,
        0, 0,
        0, 0,
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
    doc_FbxManager,
    -1,
    -1,
    0,
    0,
    init_type_FbxManager,
    0,
    0,
#if PY_MAJOR_VERSION >= 3
    0,
    0,
#else
    0,
    0,
    0,
    0,
#endif
    dealloc_FbxManager,
    0,
    0,
    0,
    release_FbxManager,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};
